# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass
import time
from collections import defaultdict

import imageio
import lpips as lpips_lib
import matplotlib
import numpy as np
import submitit
import torch
import torch.multiprocessing as mp
import wandb
from einops import rearrange
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.utils import init_data
from app.plan_common.models.wm_heads import (
    WorldModelPoseReadoutHead,
    WorldModelRewardReadoutHead,
    WorldModelViTImageHead,
)
from app.vjepa_wm.metric_names import scalar_metrics
from app.vjepa_wm.utils import (
    build_plan_eval_args,
    build_unroll_decode_eval_args,
    clean_state_dict,
    init_opt,
    init_video_model,
    load_checkpoint,
)
from app.vjepa_wm.video_wm import VideoWM
from evals.main_distributed import launch_evals_with_parsed_args as launch_evals
from src.datasets.utils.utils import get_dataset_paths
from src.utils.cluster import slurm_account_partition_and_qos
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.yaml_utils import convert_to_dict_recursive, dump_yaml, expand_env_vars

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
DEFAULT_EVAL_FREQ = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def _readable_name(value):
    """Turn config identifiers such as IsaacLiftEnvUR into stable log names."""
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def fold_views_for_image_head(image_head, video_features, visual):
    """Fold a multiview camera axis into the batch for WorldModelViTImageHead.

    The head's own view_tokens parameter is sized for exactly
    image_head.unwrapped.num_views (fixed at construction), so a decoder
    built/pretrained with num_views=1 can't consume a real view axis (e.g. a
    front_cam + wrist_cam clip encoded as b t v h w d). Mirrors
    WorldModelViTImageHead.decode()'s existing "(b v) t 1 ..." convention for
    that same mismatch, applied here to both the encoder features and the
    raw rgb target so image_head training/eval (compute_loss) can also run
    on multiview datasets, not just single-camera ones.
    """
    num_views = getattr(image_head.unwrapped, "num_views", 1)
    if video_features.ndim != 6 or video_features.shape[2] <= 1 or num_views != 1:
        return video_features, visual
    video_features = rearrange(video_features, "b t v h w d -> (b v) t 1 h w d")
    visual = rearrange(visual, "b t v c h w -> (b v) t c h w")
    return video_features, visual


def _label_comparison(array, first_label, second_label, split_axis):
    """Burn comparison labels into an HWC image or TCHW video."""
    is_video = array.ndim == 4
    frames = array.transpose(0, 2, 3, 1) if is_video else array[None]
    labeled = []
    for frame in frames:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        split = image.width // 2 if split_axis == "width" else image.height // 2
        positions = [(4, 4), (split + 4, 4)] if split_axis == "width" else [(4, 4), (4, split + 4)]
        for position, label in zip(positions, (first_label, second_label)):
            bbox = draw.textbbox(position, label)
            draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill="black")
            draw.text(position, label, fill="white")
        labeled.append(np.asarray(image))
    result = np.stack(labeled)
    return result.transpose(0, 3, 1, 2) if is_video else result[0]


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    args = expand_env_vars(args)
    folder = args.get("folder")
    checkpoint_folder = args.get("checkpoint_folder", folder)
    os.makedirs(checkpoint_folder, exist_ok=True)
    # -- META
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    load_opt_scale_epoch = cfgs_meta.get("load_opt_scale_epoch", False)
    freeze_encoder = cfgs_meta.get("freeze_encoder", True)
    r_file = cfgs_meta.get("read_checkpoint", None)
    pretrained_path = cfgs_meta.get("pretrained_path", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)

    eval_freq = cfgs_meta.get("eval_freq", DEFAULT_EVAL_FREQ)
    plan_only_eval_mode = cfgs_meta.get("plan_only_eval_mode", False)
    unroll_decode_eval_only_mode = cfgs_meta.get("unroll_decode_eval_only_mode", False)
    light_eval_only_mode = cfgs_meta.get("light_eval_only_mode", False)

    # -- LIGHT EVALS (keep as subconfigs, extract only frequently-checked flags)
    cfgs_data_traj_rollout_eval = cfgs_meta.get("data_traj_rollout_eval", {})
    cfgs_energy_landscape_eval = cfgs_meta.get("energy_landscape_eval", {})
    do_data_traj_rollout_eval = cfgs_data_traj_rollout_eval.get("do_data_traj_rollout_eval", False)
    do_energy_landscape_eval = cfgs_energy_landscape_eval.get("do_energy_landscape_eval", False)
    data_traj_decode_gt = cfgs_data_traj_rollout_eval.get("data_traj_decode_gt", False)

    light_eval_freq = cfgs_meta.get("light_eval_freq", 100)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    quick_debug = cfgs_meta.get("quick_debug", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"⚙️  Using dtype: {which_dtype}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- EVALS
    cfgs_plan_evals = args.get("evals", None)
    cfgs_unroll_decode_evals = args.get("unroll_decode_evals", None)

    # -- MODEL (extract only fields needed outside init_video_model)
    cfgs_model = args.get("model")

    # Extract heads config
    cfgs_heads = cfgs_model.get("heads_cfg", {})
    heads_architectures = cfgs_heads.get("architectures", {})
    pretrain_dec_path = cfgs_heads.get("pretrain_dec_path", None)
    new_path_heads = cfgs_heads.get("new_path_heads", {})

    # Fields needed for training loop
    rollout_cfg = cfgs_model.get("rollout_cfg", {})
    rollout_steps = rollout_cfg.get("rollout_steps", 0)
    train_rollout_prefixes = rollout_cfg.get("train_rollout_prefixes", "random")
    rollout_stop_gradient = rollout_cfg.get("rollout_stop_gradient", True)
    ctxt_window_train_rollout = rollout_cfg.get("ctxt_window_train_rollout", 8)
    do_parallel_rollout = rollout_cfg.get("do_parallel_rollout", False)
    do_sequential_rollout = rollout_cfg.get("do_sequential_rollout", True)
    sampling_rollout = rollout_cfg.get("sampling_rollout", False)
    prepend_gt_rollout_parallel = rollout_cfg.get("prepend_gt", False)

    sampling_scheduler_cfg = rollout_cfg.get("sampling_scheduler", {})
    sampling_scheduler_type = sampling_scheduler_cfg.get("type", "linear")
    sampling_scheduler_start = sampling_scheduler_cfg.get("start", 0.0)
    sampling_scheduler_end = sampling_scheduler_cfg.get("end", 0.0)

    # Derived values needed for dimensions (computed values used across the code)
    action_tokens = cfgs_model.get("action_encoder", {}).get("action_tokens", 1)
    proprio_tokens = cfgs_model.get("proprio_encoder", {}).get("proprio_tokens", 1)
    action_emb_dim = cfgs_model.get("action_encoder", {}).get("action_emb_dim", 0)
    proprio_emb_dim = cfgs_model.get("proprio_encoder", {}).get("proprio_emb_dim", 0)
    use_proprio = cfgs_model.get("use_proprio", proprio_tokens > 0 or proprio_emb_dim > 0)
    use_action = action_tokens > 0 or action_emb_dim > 0
    tubelet_size_enc = cfgs_model.get("tubelet_size_enc", 2)

    cfgs_wm_encoding = cfgs_model.get("wm_encoding", {})

    if cfgs_wm_encoding.get("dup_image", False):
        assert tubelet_size_enc == 1, "Batchify video only works with tubelet_size_enc=1"

    # -- DATA (extract only fields needed outside init_data)
    cfgs_data = args.get("data")
    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    cfgs_custom = cfgs_data.get("custom", {})
    # "camera" is the generic name for this block (camera_views, fps, etc.) --
    # historically named "droid" repo-wide even for non-DROID datasets (isaac_npz
    # lift_env included), which reads as if it only applies to DROID data. Configs
    # not yet renamed still work via the "droid" fallback.
    cfgs_camera = cfgs_data.get("camera", cfgs_data.get("droid", {}))
    # Predictor attention (causal mask sizing, RoPE) is built once at model
    # construction time, so the view count has to be static per run: it must
    # match len(camera_views) exactly, for both the train and validation
    # dataset (val_dataset_camera_views), since both share one model.
    _camera_views = cfgs_camera.get("camera_views", ["front_cam"])
    num_views = len(_camera_views) if isinstance(_camera_views, (list, tuple)) else 1
    _val_camera_views = cfgs_validation.get("val_dataset_camera_views", _camera_views)
    if isinstance(_val_camera_views, (list, tuple)) and len(_val_camera_views) != num_views:
        raise ValueError(
            f"camera_views has {num_views} view(s) but val_dataset_camera_views has "
            f"{len(_val_camera_views)}; the predictor's attention is built for a fixed "
            "view count shared by train and validation."
        )

    # Compute dataset paths
    datasets = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(datasets), "Must have one sampling weight specified for each dataset"

    dataset_type = cfgs_data.get("dataset_type", "custom")
    if dataset_type.lower() == "mixed_dataset":
        dataset_paths = datasets
    else:
        dataset_paths = get_dataset_paths(datasets)

    val_datasets = cfgs_validation.get("val_datasets", [])
    val_dataset_paths = get_dataset_paths(val_datasets) if val_datasets else None

    # val_datasets_1..4 subconfigs: extra eval/viz loaders beyond the primary
    # combined one, e.g. embodiment-pure validation loaders or (with
    # split: train) a demo-type-filtered slice of the actual training data.
    # Each can override image_head_key so its rollout decodes/scores through
    # a different heads[] decoder than the default "image_head" (see
    # val_loader_head_keys below). val_datasets_1 alone is the long-standing
    # slot other configs already set (usually to null); _2/_3/_4 are new.
    extra_val_slots = {}
    for slot_num in (1, 2, 3, 4):
        slot_cfg = cfgs_validation.get(f"val_datasets_{slot_num}", None)
        if slot_cfg is not None:
            extra_val_slots[slot_num] = (slot_cfg, get_dataset_paths(slot_cfg.get("names")))

    # Fields used outside init_data
    frameskip = cfgs_custom.get("frameskip", True)
    action_skip = cfgs_custom.get("action_skip", 1)
    state_skip = cfgs_custom.get("state_skip", 1)
    img_size = cfgs_data.get("img_size", 256)
    num_workers = cfgs_loader.get("num_workers", 1)
    filter_first_episodes = cfgs_custom.get("filter_first_episodes", None)
    val_viz_rank0_loader = cfgs_validation.get("val_viz_rank0_loader", False)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")

    # -- LOSS
    cfgs_loss = args.get("loss")

    # -- OPTIMIZATION (simplify main_optimizer logic)
    cfgs_opt = args.get("optimization")
    train_heads = cfgs_opt["train_heads"]

    main_optimizer = cfgs_opt["main_optimizer"]
    if main_optimizer == "transition_model":
        num_epochs = cfgs_opt["transition_model"]["num_epochs"]
        ipe = cfgs_opt["transition_model"]["iterations_per_epoch"]
        train_predictor = True
        train_heads_on_predictor = False
    else:  # image_head | state_head | reward_head
        num_epochs = cfgs_opt["heads"][main_optimizer]["num_epochs"]
        ipe = cfgs_opt["heads"][main_optimizer]["iterations_per_epoch"]
        train_predictor = cfgs_opt["heads"]["train_predictor"]
        train_heads_on_predictor = cfgs_opt["heads"]["train_heads_on_predictor"]

    # -- LOGGING
    cfgs_logging = args.get("logging")
    tag = cfgs_logging.get("write_tag", "jepa")
    latest_format = cfgs_logging.get("latest_format", "pth.tar")
    cfgs_wandb = cfgs_logging.get("wandb")

    if light_eval_only_mode:
        light_eval_freq = 1
    if plan_only_eval_mode or light_eval_only_mode or quick_debug or unroll_decode_eval_only_mode:
        filter_first_episodes, num_workers = 10, 0

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"🚀 Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    train_log_file = os.path.join(folder, f"log_r{rank}.csv")
    pref_tag = f"{tag}-" if tag else ""
    latest_file = pref_tag + f"latest.{latest_format}"
    latest_path = os.path.join(checkpoint_folder, latest_file)
    finetune = pretrained_path is not None
    logger.info(f"{'🔧 Finetuning mode' if finetune else '🆕 Training from scratch'}")
    # finetune covers the case of training heads on top of frozen transition_model
    # if cfgs_model["pretrained_path"] is None, i.e. training head just on encoder
    head_training_mode = main_optimizer in ["image_head", "state_head"]
    resume = os.path.exists(latest_path)
    resume_finetune = os.path.exists(latest_path) and finetune
    resume_latest = os.path.exists(latest_path) and not finetune
    if resume_finetune:
        logger.info("♻️  Resuming from checkpoint")
    load_path = None
    if load_model:
        if r_file is not None:
            # An explicit checkpoint was requested (meta.read_checkpoint, e.g.
            # via --resume-run --resume-checkpoint) -- honor it regardless of
            # finetune/"-latest" state, so resuming isn't limited to whichever
            # file happens to be named "-latest".
            load_path = r_file if os.path.isabs(r_file) else os.path.join(folder, r_file)
            load_opt_scale_epoch = not head_training_mode
            logger.info(f"♻️  Resuming from explicit checkpoint: {load_path}")
        elif resume_finetune:
            load_path = latest_path
            load_opt_scale_epoch = not head_training_mode
        elif resume_latest:
            load_path = latest_path
            load_opt_scale_epoch = not head_training_mode
        else:  # not resuming, i.e. not os.path.exists(latest_path)
            load_path = pretrained_path
        if load_path is None or not os.path.exists(load_path):
            load_path = None
            load_model = False

    train_csv_logger, eval_csv_logger = None, None

    def create_csv_logger(losses, total_stats, train=False):
        if train:
            csv_log_file = train_log_file
        else:
            if light_eval_only_mode:
                csv_log_file = os.path.join(folder, f"light_eval_only_eval_r{rank}.csv")
            else:
                csv_log_file = os.path.join(folder, f"eval_r{rank}.csv")
        # Get all unique keys from eval_losses and eval_total_stats
        excluded_keys = [
            "eval_data/image_rollouts",
            "eval_data/image_rollouts_noisy_actions",
            "eval_data/image_animated_rollout",
        ]
        all_keys = {key for key in list(losses.keys()) + list(total_stats.keys()) if key not in excluded_keys}
        # Sort keys lexicographically for consistent order
        sorted_keys = sorted(all_keys)
        # Create a format tuple for each key
        new_columns = [("%.5f", key) for key in sorted_keys]
        # Initialize the logger with new columns
        if train:
            global train_csv_logger_columns
            train_csv_logger_columns = ["epoch", "itr", "loss", "gpu-time(ms)", "iter-time(ms)"] + sorted_keys
            # Header must declare the same 5 leading columns log_stats() actually
            # writes (epoch, itr, loss, gpu-time(ms), iter-time(ms)) -- previously
            # only epoch/itr were declared here, so CSVLogger.log()'s zip(types,
            # data) silently truncated to the shorter list, shifting every logged
            # value 3 columns left of its true header name and dropping the last
            # 3 sorted_keys entirely.
            return CSVLogger(
                csv_log_file,
                ("%d", "epoch"),
                ("%d", "itr"),
                ("%.5f", "loss"),
                ("%.5f", "gpu-time(ms)"),
                ("%.5f", "iter-time(ms)"),
                *new_columns,
            )
        else:
            global eval_csv_logger_columns
            eval_csv_logger_columns = ["epoch", "itr"] + sorted_keys
        return CSVLogger(csv_log_file, ("%d", "epoch"), ("%d", "itr"), *new_columns)

    # -- init data-loaders/samplers
    transform = make_transforms(
        img_size=img_size,
        **cfgs_data_aug,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)

    # Prepare data kwargs from config, flattening nested structures and filtering out non-init_data fields
    excluded_keys = [
        "datasets",
        "val_datasets",
        "img_size",
        # subfields of cfgs_data
        "validation",
        "loader",
        "custom",
        "droid",
    ]
    data_kwargs = {k: v for k, v in cfgs_data.items() if k not in excluded_keys}
    # Add fields from nested structures
    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_camera)
    # Add computed/override parameters
    data_kwargs.update(
        {
            "data_paths": dataset_paths,
            "val_data_paths": val_dataset_paths,
            "transform": transform,
            "world_size": world_size,
            "rank": rank,
            # overridden by quick_debug
            "filter_first_episodes": filter_first_episodes,  # Potentially overridden by mode
            "num_workers": num_workers,
        }
    )

    val_data_iters = []
    primary_val_names = val_datasets or cfgs_data.get("datasets", [])
    val_loader_names = [_readable_name("_".join(primary_val_names)) or "validation"]
    # Parallel to val_loader_names: which heads[] image decoder each loader's
    # eval/rollout should decode through. Defaults to "image_head" everywhere
    # (old behavior); val_datasets_1/_2 can override via image_head_key so an
    # embodiment-pure loader uses that embodiment's own finetuned decoder
    # instead of the one global default (see step_model's head_key param).
    val_loader_head_keys = ["image_head"]
    (
        dataset,
        val_dataset,
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
        viz_val_data_loader,
    ) = init_data(**data_kwargs)
    val_data_iters.append((val_dataset, val_traj_dataset, val_unsupervised_loader))

    # val_datasets_1..4: each makes its own init_data call over its own
    # data_paths (so it gets its own internal train/valid split, independent
    # of the primary loader's). split: "valid" (default) uses that call's
    # held-out slice, as before; split: "train" uses its *train* slice
    # instead -- e.g. to visualize the actual training distribution
    # restricted to success/failure via filter_train_by_demo_types, rather
    # than a held-out set. Either way the result is just one more entry in
    # val_data_iters/val_loader_names/val_loader_head_keys; step_model
    # doesn't distinguish "train-sourced" loaders from real validation ones.
    for slot_num, (slot_cfg, slot_paths) in sorted(extra_val_slots.items()):
        split = slot_cfg.get("split", "valid")
        if split not in ("train", "valid"):
            raise ValueError(f"val_datasets_{slot_num}.split must be 'train' or 'valid', got {split!r}")
        data_kwargs_slot = data_kwargs.copy()
        data_kwargs_slot.update(
            {
                "dset_fraction": 1,
                "val_dset_fraction": 1,
                "data_paths": slot_paths,
                "val_data_paths": slot_paths,
                "batch_size": slot_cfg.get("batch_size", 4),
                "drop_last": slot_cfg.get("drop_last", True),
                "fps": slot_cfg.get("fps", 4),
                "dataset_fpcs": slot_cfg.get("fpcs", [8]),
                "val_dataset_fpcs": slot_cfg.get("fpcs", [8]),
                "camera_views": slot_cfg.get("camera_views", ["exterior_image_2_left"]),
                "droid_to_rcasa_action_format": slot_cfg.get("droid_to_rcasa_action_format", 1),
                "filter_train_by_demo_types": split == "train",
            }
        )
        (
            slot_train_dataset,
            slot_val_dataset,
            slot_train_traj_dataset,
            slot_val_traj_dataset,
            slot_train_loader,
            slot_val_loader,
            _,
            slot_viz_val_loader,
        ) = init_data(**data_kwargs_slot)
        if split == "train":
            val_data_iters.append((slot_train_dataset, slot_train_traj_dataset, slot_train_loader))
        else:
            val_data_iters.append((slot_val_dataset, slot_val_traj_dataset, slot_val_loader))
            if slot_num == 1:
                # Preserve old behavior: val_datasets_1 alone used to also
                # replace the rank0 viz loader. Later slots don't touch it.
                viz_val_data_loader = slot_viz_val_loader
        name_suffix = "_train" if split == "train" else ""
        default_name = f"validation_{slot_num + 1}{name_suffix}"
        slot_name = _readable_name("_".join(slot_cfg.get("names", [])))
        val_loader_names.append(f"{slot_name}{name_suffix}" if slot_name else default_name)
        val_loader_head_keys.append(slot_cfg.get("image_head_key", "image_head"))

    if dataset_type in ("custom", "isaac_npz") and traj_dataset is not None:
        preprocessor = Preprocessor(
            action_mean=traj_dataset.action_mean,
            action_std=traj_dataset.action_std,
            state_mean=traj_dataset.state_mean,
            state_std=traj_dataset.state_std,
            proprio_mean=traj_dataset.proprio_mean,
            proprio_std=traj_dataset.proprio_std,
            transform=transform,
            inverse_transform=inverse_transform,
        )
    else:
        preprocessor = None

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"📊 Iterations per epoch: {ipe} (dataset size: {_dlen})")
    if main_optimizer == "transition_model":
        cfgs_opt["transition_model"]["iterations_per_epoch"] = ipe
    elif main_optimizer == "image_head":
        cfgs_opt["heads"]["image_head"]["iterations_per_epoch"] = ipe
    elif main_optimizer == "state_head":
        cfgs_opt["heads"]["state_head"]["iterations_per_epoch"] = ipe

    # Logger
    class Trainer:
        def __init__(self, config):
            if quick_debug:
                config["debug"] = quick_debug
            self.config = config
            self.ipe = ipe
            self.use_wandb = config.get("use_wandb", False)
            self.disable_wandb_media = config.get("disable_wandb_media", False)
            self.log_media_locally = config.get("log_media_locally", False)
            self.local_log_dir = None
            if self.log_media_locally and rank == 0:
                self.local_log_dir = os.path.join(folder, "local_logs")
                os.makedirs(self.local_log_dir, exist_ok=True)
            if self.use_wandb and rank == 0:
                project_name = config.get("project", "vjepa_wm") if not config["debug"] else "vjepa_wm_debug"
                wandb_run_id_file = os.path.join(folder, "wandb_run_id.txt")
                if os.path.exists(wandb_run_id_file):
                    with open(wandb_run_id_file, "r") as f:
                        wandb_run_id = f.read().strip()
                    wandb.init(project=project_name, id=wandb_run_id, resume="allow", dir=folder, config=args)
                    logger.info(f"Resuming Wandb run {wandb_run_id}")
                else:
                    wandb.init(project=project_name, dir=folder, config=args)
                    with open(wandb_run_id_file, "w") as f:
                        f.write(wandb.run.id)
                wandb.run.name = config.get("run_name", os.path.basename(folder))
                self.job_set = set()
                # Also upload the exact resolved config file (dumped by app/main.py) so
                # the run's Files tab carries the literal yaml alongside the searchable
                # config panel populated above.
                params_path = os.path.join(folder, "params-pretrain.yaml")
                if os.path.exists(params_path):
                    wandb.save(params_path, policy="now")
                else:
                    logger.warning(f"Expected config file not found, skipping wandb upload: {params_path}")

        def log(self, epoch, itr, losses, total_stats, eval_losses=None, eval_total_stats=None, image_stats=None):
            log_dict = {
                "epoch": epoch + 1,
                "itr": itr,
            }
            train_metrics = {**losses, **total_stats}
            eval_metrics = {**(eval_losses or {}), **(eval_total_stats or {})}
            for split, metrics in (("train", train_metrics), ("eval", eval_metrics)):
                named = scalar_metrics(metrics, split, cfgs_loss)
                for key, value in named.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    log_dict[key] = value
            if image_stats:  # not None or nonempty
                if self.log_media_locally and rank == 0:
                    self.log_media_local(image_stats, epoch, itr)
                if not self.disable_wandb_media:
                    log_dict.update(image_stats)
            # Console output is a single per-epoch tqdm progress bar (see the
            # training loop) instead of a "[epoch, itr] loss: ..." line per
            # log_freq iters; wandb/CSV logging below is unaffected.
            if self.use_wandb and rank == 0:
                wandb.log(log_dict)

        def log_media_local(self, image_stats, epoch=None, itr=None):
            step = epoch * ipe + itr
            for key, value in image_stats.items():
                subfolder = os.path.join(self.local_log_dir, "/".join(key.split("/")))
                os.makedirs(subfolder, exist_ok=True)
                filename = f"{step}.gif" if isinstance(value, wandb.Video) else f"{step}.pdf"
                if isinstance(value, wandb.Video):
                    frames = value._prepare_video(value.data)
                    duration = 1.0 / 10  # assuming 10 FPS
                    imageio.mimsave(os.path.join(subfolder, filename), frames, duration=duration, loop=0)
                elif isinstance(value, wandb.Image):
                    value.image.save(os.path.join(subfolder, filename))
                elif isinstance(value, matplotlib.figure.Figure):
                    value.savefig(os.path.join(subfolder, filename), bbox_inches=None)

    trainer = Trainer(cfgs_wandb)

    # -- init model
    if use_action:
        actions_per_vid_feat = tubelet_size_enc * frameskip // action_skip
        model_action_dim = traj_dataset.action_dim * tubelet_size_enc * frameskip // action_skip
    else:
        actions_per_vid_feat, model_action_dim = None, None
    if use_proprio:
        proprio_multiplier = tubelet_size_enc * frameskip // state_skip
        model_proprio_dim = traj_dataset.proprio_dim * tubelet_size_enc // state_skip
    else:
        proprio_multiplier, model_proprio_dim = None, None

    # Prepare model kwargs by flattening nested configs and filtering out non-init_video_model fields
    excluded_keys = [
        "rollout_cfg",
        "heads_cfg",
        "pretrained_path",
        "visual_encoder",
        "action_encoder",
        "proprio_encoder",
        "predictor",
        "wm_encoding",
        "attn",
    ]
    model_kwargs = {k: v for k, v in cfgs_model.items() if k not in excluded_keys}
    if "visual_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["visual_encoder"])
    if "action_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["action_encoder"])
    if "proprio_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["proprio_encoder"])
    if "predictor" in cfgs_model:
        model_kwargs.update(cfgs_model["predictor"])
    model_kwargs.update(
        {
            "device": device,
            "img_size": img_size,
            "action_dim": model_action_dim,  # Computed from dataset
            "proprio_dim": model_proprio_dim,  # Computed from dataset
            "cfgs_attn_pattern": cfgs_model.get("attn", None),  # Pass attn subconfig
            "use_proprio": use_proprio,  # Computed derived value
            "use_action": use_action,  # Computed derived value
            "num_views": num_views,  # Computed from data.droid.camera_views
        }
    )
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(**model_kwargs)
    if predictor is not None and hasattr(predictor, "action_encoder_output_dim"):
        action_dimensions = {
            "action_dim": model_action_dim,
            "action_encoder_output_dim": predictor.action_encoder_output_dim,
            "adaln_conditioning_dim": predictor.predictor_total_embed_dim,
            "use_proprio": use_proprio,
        }
        logger.info(f"Action conditioning dimensions: {action_dimensions}")
        if rank == 0 and wandb.run is not None:
            wandb.run.summary.update(action_dimensions)

    heads = {}
    if train_heads or pretrain_dec_path is not None:
        # "image_head" plus any "image_head_<suffix>" entries (e.g. per-embodiment
        # decoders such as "image_head_ur") -- all built the same way, each kept
        # under its own heads[] key so eval can pick the right one per batch.
        for head_name in heads_architectures:
            if head_name != "image_head" and not head_name.startswith("image_head_"):
                continue
            image_head_type = heads_architectures[head_name]["kind"]
            if image_head_type is not None and image_head_type.lower() != "none":
                if image_head_type == "vit":
                    decoder = WorldModelViTImageHead(
                        head_config=dict(heads_architectures[head_name]["config"]),
                        inverse_transform=inverse_transform,
                        device=device,
                    )
                heads[head_name] = decoder
        if "state_head" in heads_architectures:
            state_decoder = WorldModelPoseReadoutHead(
                head_config=dict(heads_architectures["state_head"]["config"]), device=device
            )
            heads["state_head"] = state_decoder
        if "reward_head" in heads_architectures:
            reward_decoder = WorldModelRewardReadoutHead(
                head_config=dict(heads_architectures["reward_head"]["config"]), device=device
            )
            heads["reward_head"] = reward_decoder

    # -- init optimizer and scheduler
    if train_predictor and predictor is not None:
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            encoder=encoder,
            freeze_encoder=freeze_encoder,
            **cfgs_opt["transition_model"],
        )
        clip_grad = cfgs_opt["transition_model"]["clip_grad"]
        use_radamw = cfgs_opt["transition_model"]["use_radamw"]
        if sampling_scheduler_type == "linear":
            # linear decay from sampling_scheduler_start to sampling_scheduler_end
            rollout_sampling_scheduler = (
                sampling_scheduler_start - i * (sampling_scheduler_start - sampling_scheduler_end) / (ipe * num_epochs)
                for i in range(int(ipe * num_epochs) + 1)
            )
        elif sampling_scheduler_type == "exponential":
            # exponential decay from sampling_scheduler_start to sampling_scheduler_end
            rollout_sampling_scheduler = (
                sampling_scheduler_start
                * (sampling_scheduler_end / sampling_scheduler_start) ** (i / (ipe * num_epochs))
                for i in range(int(ipe * num_epochs) + 1)
            )
        elif sampling_scheduler_type == "sigmoid":
            rollout_sampling_scheduler = (
                sampling_scheduler_start
                + (sampling_scheduler_end - sampling_scheduler_start)
                * (1 / (1 + np.exp(-10 * (i / (ipe * num_epochs) - 0.5))))
                for i in range(int(ipe * num_epochs) + 1)
            )
    else:
        optimizer, scaler, scheduler, wd_scheduler, clip_grad, use_radamw = None, None, None, None, None, None
    if train_heads:
        for name, head in heads.items():
            head_opt_cfg = dict(cfgs_opt["heads"][name])
            # Joint transition/head runs inherit the actual loader length.
            if head_opt_cfg.get("iterations_per_epoch") is None:
                head_opt_cfg["iterations_per_epoch"] = ipe
            head.init_opt(**head_opt_cfg)

    start_epoch = 0
    resumed_heads = False
    # -- load training checkpoint
    if load_model:
        # to resume predictor or head training
        expected_head_paths = [
            load_path.removesuffix(".pth.tar") + f"_{name}.pth.tar" for name in heads
        ]
        load_heads = bool(heads) and resume and all(os.path.exists(path) for path in expected_head_paths)
        resumed_heads = load_heads
        logger.info(f"Load heads: {load_heads}")
        (
            predictor,
            action_encoder,
            proprio_encoder,
            heads,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            heads=heads,
            opt=optimizer,
            scaler=scaler,
            load_opt_scale_epoch=load_opt_scale_epoch,
            load_heads=load_heads,
            train_heads=train_heads,
            train_predictor=train_predictor,
        )
        # Only resume the schedulers if we resume a pretraining or a finetuning
        # Not if we start a finetuning: we reset them
        if load_opt_scale_epoch and scheduler is not None and wd_scheduler is not None:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
        if light_eval_only_mode:
            start_epoch -= 1

    # Load pretrained heads from pretrain_dec_path
    if pretrain_dec_path is not None and not resumed_heads:
        for name, head in heads.items():
            if new_path_heads.get(name, True):
                head_path = pretrain_dec_path[name].removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.load_checkpoint(head_path)
                logger.info(f"loaded pretrained head named {name}")
            else:
                head_path = pretrain_dec_path[name]
                if isinstance(head_path, str) and head_path.startswith(("http://", "https://")):
                    epoch = head.load_checkpoint(head_path, load_optimizer=False)
                    logger.info(f"loaded pretrained head named {name} from epoch {epoch}")
                else:
                    checkpoint = torch.load(head_path, map_location=torch.device("cpu"))
                    if "model" in checkpoint:
                        epoch = head.load_checkpoint(head_path, load_optimizer=False)
                        logger.info(f"loaded pretrained head named {name} from epoch {epoch}")
                    else:
                        epoch = checkpoint["epoch"]
                        pretrained_dict = clean_state_dict(checkpoint[name])
                        msg = head.model.load_state_dict(pretrained_dict, strict=False)
                        logger.info(f"loaded pretrained head named {name} from epoch {epoch} with msg: {msg}")
                    del checkpoint

    # DDP wrapping after loading state_dicts
    if not freeze_encoder:
        encoder = DDP(encoder, static_graph=False, find_unused_parameters=False)
    if train_predictor:
        if action_encoder is not None:
            action_encoder = DDP(action_encoder, static_graph=False, find_unused_parameters=False)
        if proprio_encoder is not None:
            proprio_encoder = DDP(proprio_encoder, static_graph=False, find_unused_parameters=False)
        predictor = DDP(predictor, static_graph=False, find_unused_parameters=False)
    for name in heads.keys():
        heads[name].model = DDP(heads[name].model, static_graph=False, find_unused_parameters=False)

    # Prepare VideoWM kwargs from config
    wm_kwargs = {
        "device": device,
        # Model components
        "encoder": encoder,
        "predictor": predictor,
        "action_encoder": action_encoder,
        "proprio_encoder": proprio_encoder,
        # Computed dimensions
        "action_dim": model_action_dim,
        "proprio_dim": model_proprio_dim,
        "use_proprio": use_proprio,
        "use_action": use_action,
        # From cfgs_model (pass directly from config)
        "action_tokens": action_tokens,
        "proprio_tokens": proprio_tokens,
        "num_views": num_views,  # Computed from data.droid.camera_views
        "grid_size": cfgs_model.get("grid_size", 14),
        "tubelet_size_enc": cfgs_model.get("tubelet_size_enc", 2),
        "action_conditioning": cfgs_model.get("action_conditioning", "token"),
        "proprio_encoding": cfgs_model.get("proprio_encoding", "feature"),
        "enc_type": cfgs_model["visual_encoder"].get("enc_type", "vjepa"),
        "pred_type": cfgs_model["predictor"].get("pred_type", "dino_wm"),
        "action_encoder_inpred": cfgs_model["action_encoder"].get("action_encoder_inpred", False),
        "proprio_encoder_inpred": cfgs_model["proprio_encoder"].get("proprio_encoder_inpred", False),
        # Previously only reached AdaLN (which reads it directly in its own
        # config path); apply it uniformly here so DINO-WM and other
        # pred_types also normalize action conditioning when configured to.
        "normalize_action_conditioning": cfgs_model.get("predictor", {}).get("normalize_action_conditioning", False),
        **cfgs_wm_encoding,
        # From cfgs_data
        "action_skip": cfgs_data.get("action_skip", 1),
        "frameskip": cfgs_data.get("frameskip", 1),
        "img_size": cfgs_data.get("img_size", 256),
        # Heads
        "heads": heads,
        # Optimization
        "scaler": scaler,
        "optimizer": optimizer,
        "clip_grad": clip_grad,
        "mixed_precision": mixed_precision,
        "use_radamw": use_radamw,
        # Loss config (pass subconfig directly)
        "cfgs_loss": cfgs_loss,
    }
    world_model = VideoWM(**wm_kwargs)

    # -- Initialize LPIPS once for evaluation
    lpips = lpips_lib.LPIPS(net="vgg").eval().to(device)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "predictor": world_model.predictor.state_dict() if world_model.predictor is not None else None,
            "opt": optimizer.state_dict() if optimizer is not None else None,
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
        }
        if world_model.action_encoder is not None and not cfgs_model["action_encoder"].get(
            "action_encoder_inpred", False
        ):
            save_dict.update({"action_encoder": world_model.action_encoder.state_dict()})
        if (
            world_model.proprio_encoder is not None
            and use_proprio
            and not cfgs_model["proprio_encoder"].get("proprio_encoder_inpred", False)
        ):
            save_dict.update({"proprio_encoder": world_model.proprio_encoder.state_dict()})
        if train_heads:
            for name, head in world_model.heads.items():
                head_path = path.removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.save_checkpoint(epoch, head_path)
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    train_loader = iter(unsupervised_loader)
    if val_data_iters != [(None, None, None)]:
        val_loader_iters = []
        for vl_dset, vl_traj_dset, vl_loader in val_data_iters:
            val_loader_iters.append(iter(vl_loader))

    if viz_val_data_loader is not None:
        viz_val_loader_iter = iter(viz_val_data_loader)

    def load_clips(sample):
        all_clips = []
        return sample[0][0].to(device, non_blocking=True), None, None

    def get_batch(train=True, idx=0):
        nonlocal train_loader, val_loader_iters
        if dataset_type in ("custom", "isaac_npz"):
            try:
                if train:
                    obs, action, state, reward = next(train_loader)
                else:
                    obs, action, state, reward = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info(f"Exhausted data loaders idx {idx} with {train=}. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    obs, action, state, reward = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    obs, action, state, reward = next(val_loader_iters[idx])
            # Multiview batches (obs["visual"]: b t v c h w) now flow through
            # VideoWM.encode_obs unchanged -- each view is encoded separately
            # and concatenated into the predictor's token sequence, rather
            # than being split into independent single-view samples here.
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=dtype, non_blocking=True)
            action = action.to(device, dtype=dtype, non_blocking=True)
            state = state.to(device, dtype=dtype, non_blocking=True)
            reward = reward.to(device, dtype=dtype, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                if train:
                    sample = next(train_loader)
                else:
                    sample = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted data loaders. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    sample = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    sample = next(val_loader_iters[idx])
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    def get_viz_batch():
        """Get a visualization batch from the non-distributed validation loader (rank 0 only)"""
        nonlocal viz_val_loader_iter
        if rank != 0 or viz_val_loader_iter is None:
            return None, None, None, None, None, None
        if dataset_type in ("custom", "isaac_npz"):
            try:
                obs, action, state, reward = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                obs, action, state, reward = next(viz_val_loader_iter)
            # Multiview batches (obs["visual"]: b t v c h w) now flow through
            # VideoWM.encode_obs unchanged -- each view is encoded separately
            # and concatenated into the predictor's token sequence, rather
            # than being split into independent single-view samples here.
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=torch.float32, non_blocking=True)
            action = action.to(device, dtype=torch.float32, non_blocking=True)
            state = state.to(device, dtype=torch.float32, non_blocking=True)
            reward = reward.to(device, dtype=torch.float32, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                sample = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                sample = next(viz_val_loader_iter)
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    # -- TRAINING LOOP
    if not (plan_only_eval_mode or unroll_decode_eval_only_mode):
        for epoch in range(start_epoch, num_epochs):
            logger.info("\n" + "─" * 50)
            logger.info(f"📈 Epoch {epoch + 1}/{num_epochs}")
            logger.info("─" * 50)

            # -- update distributed-data-loader epoch
            unsupervised_sampler.set_epoch(epoch)

            loss_meter = AverageMeter()
            gpu_time_meter = AverageMeter()
            wall_time_meter = AverageMeter()

            epoch_pbar = tqdm(range(ipe), desc=f"Epoch {epoch + 1}/{num_epochs}", disable=(rank != 0))
            for itr in epoch_pbar:
                itr_start_time = time.time()
                if quick_debug or light_eval_only_mode:
                    if itr > 5:
                        break

                def step_model(obs, action, state, reward, train=True, head_key="image_head"):
                    # Eval-only: which image_head entry decodes/scores this batch's rollout.
                    # Lets embodiment-pure validation loaders (see val_loader_head_keys) use
                    # their own finetuned decoder instead of the single default "image_head".
                    # Falls back to "image_head" if head_key isn't a loaded head (e.g. legacy
                    # configs with only one decoder, or train_heads is False and it never loaded).
                    image_head_name = head_key if head_key in world_model.heads else "image_head"
                    rates = defaultdict(float)
                    if train:
                        if train_predictor:
                            rates["info/transition_model/lr"] = scheduler.step()
                            rates["info/transition_model/wd"] = wd_scheduler.step()
                        if train_heads:
                            for name, head in world_model.heads.items():
                                rates[f"info/{name}/lr"] = head.scheduler.step()
                                rates[f"info/{name}/wd"] = head.wd_scheduler.step()
                    else:
                        rates["info/transition_model/lr"] = 0.0
                        rates["info/transition_model/wd"] = 0.0
                        for name, head in world_model.heads.items():
                            rates[f"info/{name}/lr"] = 0.0
                            rates[f"info/{name}/wd"] = 0.0
                    # --

                    # Step 1. Forward
                    total_stats = {}
                    # Initialized here (rather than only at the rollout-eval section
                    # below) so the image_head block (2) can also populate it on
                    # eval steps -- it runs before that section.
                    image_stats = {}
                    total_transition_loss = 0.0
                    total_head_loss = 0.0
                    if action is not None:
                        total_stats.update(
                            {
                                "act_mean": action.mean(),
                                "act_std": action.std(),
                                "act_min": action.min(),
                                "act_max": action.max(),
                            }
                        )
                    # 1. TRAIN PREDICTOR TO PREDICT ONE STEP IN THE FUTURE USING TEACHER FORCING
                    train_rollout_result = {}
                    parallel_rollout_result = {}
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        video_features, proprio_features, action_features = world_model.encode(obs, action)
                        if train_predictor and train and predictor is not None:
                            pred_video_features, pred_action_features, pred_proprio_features = (
                                world_model.forward_pred(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                )
                            )
                            predictor_losses = world_model.compute_loss(
                                pred_video_features,
                                pred_proprio_features,
                                video_features,
                                # video_features comes from the frozen, shared encoder so it's
                                # already a fixed target; proprio_features comes from the
                                # trainable proprio encoder and must be detached here too, or
                                # this loss can move the target instead of just the prediction
                                # (rollout() already detaches its proprio targets, see below).
                                proprio_features.detach() if proprio_features is not None else None,
                                shift=1,
                            )
                            with torch.no_grad():
                                copy_mse = (video_features[:, 1:].float() - video_features[:, :-1].float()).square().mean()
                                pred_mse = (pred_video_features[:, :-1].float() - video_features[:, 1:].float()).square().mean()
                                total_stats["copy_previous_latent_mse"] = copy_mse.item()
                                total_stats["prediction_vs_copy_mse_ratio"] = (pred_mse / copy_mse.clamp_min(1e-8)).item()
                        else:
                            pred_video_features, pred_proprio_features = None, None
                            predictor_losses = {}
                    # Weight every prediction horizon equally: rollout_steps is the total
                    # number of predicted horizons (1 teacher-forced step, plus
                    # rollout_steps - 1 further rollout steps below), each of which should
                    # get weight 1/rollout_steps. rollout() below is called with
                    # rollout_steps - 1 and independently divides by (that + 1), so it
                    # already lands on 1/rollout_steps per extra step -- this just makes the
                    # teacher-forced term match instead of the old 1/(rollout_steps + 1),
                    # which over-weighted rollout steps relative to the teacher-forced one
                    # (and halved the teacher-forced loss even when rollout_steps == 1 and no
                    # extra rollout ran at all).
                    predictor_loss = predictor_losses.get("loss", 0.0) / max(rollout_steps, 1)
                    if train and train_predictor and predictor is not None:
                        total_transition_loss += predictor_loss
                    stats = defaultdict(list)
                    for k in predictor_losses:
                        val = predictor_losses[k]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().clone()
                        else:
                            val = torch.tensor(val)
                        stats[k].append(val.unsqueeze(0))
                    # mean over 1 element in list
                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                    for k in stats:
                        for j in range(len(stats[k])):
                            train_rollout_result[f"train_rollout/{k}/{j+1}"] = stats[k][j].item()
                    # 2. TRAIN HEADS ON FEATURES FROM ENCODER
                    if "image_head" in world_model.heads and train_heads:
                        image_head = world_model.heads["image_head"]
                        head_video_features, head_visual = fold_views_for_image_head(
                            image_head, video_features, obs["visual"]
                        )
                        # Encoder produces features of videos normalized by mean and std
                        # So let's keep them for target of decoder loss
                        target_rgb_video = image_head.preprocess_rgb(head_visual[:, ::tubelet_size_enc])
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_image_losses = image_head.compute_loss(
                                head_video_features.detach(),
                                target_rgb_video,
                                global_step=epoch * ipe + itr,
                            )
                        rates["info/image_head/examples_seen"] += (
                            head_video_features.shape[0] * head_video_features.shape[1]
                        )
                        encoder_image_losses = {k: v.mean() for k, v in encoder_image_losses.items()}
                        loss = encoder_image_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["image_head"].backward(loss)
                        encoder_image_losses = {
                            "encoder_image_" + k: v.item() for k, v in encoder_image_losses.items()
                        }
                        total_stats.update(encoder_image_losses)
                        # Ground-truth vs. predicted image, logged on eval steps only
                        # (this is the standalone decoder-training config's only
                        # visualization -- do_data_traj_rollout_eval's rollout
                        # images don't apply here since there's no predictor).
                        if not train and rank == 0:
                            # fold_views_for_image_head merges "b t v ... -> (b v) t ..." with
                            # v as the fast axis, so sample 0's views sit at merged indices
                            # 0..num_views-1 (view 0 = front_cam, view 1 = wrist_cam, per
                            # data.droid.camera_views) -- loop over them instead of only ever
                            # reading merged index 0 (sample 0's front_cam, wrist_cam skipped).
                            view_names = _camera_views if len(_camera_views) == num_views else range(num_views)
                            with torch.no_grad():
                                for v, view_name in enumerate(view_names):
                                    pred_rgb = image_head.decode(head_video_features[v : v + 1, :1])[0, 0, 0]  # h w c, uint8
                                    # inverse_transform's mean/std buffers are plain (undeviced)
                                    # tensors -- postprocess_rgb() also .cpu()'s its input before
                                    # calling it, for the same reason.
                                    gt_rgb = image_head.inverse_transform(head_visual[v : v + 1, :1].cpu())[0, 0]  # c h w
                                    gt_rgb = (255.0 * gt_rgb).clip(0.0, 255.0).to(torch.uint8).permute(1, 2, 0)
                                    image_stats[f"eval_image/{view_name}/ground_truth"] = wandb.Image(gt_rgb.cpu().numpy())
                                    image_stats[f"eval_image/{view_name}/predicted"] = wandb.Image(pred_rgb.cpu().numpy())
                    if "state_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_state_losses = world_model.heads["state_head"].compute_loss(
                                video_features.detach(), None, state[:, ::tubelet_size_enc]
                            )
                        rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_state_losses = {k: v.mean() for k, v in encoder_state_losses.items()}
                        loss = encoder_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["state_head"].backward(loss)
                        encoder_state_losses = {
                            "encoder_state_" + k: v.item() for k, v in encoder_state_losses.items()
                        }
                        total_stats.update(encoder_state_losses)
                    if "reward_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_reward_losses = world_model.heads["reward_head"].compute_loss(
                                video_features.detach(), reward[:, ::tubelet_size_enc]
                            )
                        rates["info/reward_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_reward_losses = {k: v.mean() for k, v in encoder_reward_losses.items()}
                        loss = encoder_reward_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["reward_head"].backward(loss)
                        encoder_reward_losses = {
                            "encoder_reward_" + k: v.item() for k, v in encoder_reward_losses.items()
                        }
                        total_stats.update(encoder_reward_losses)
                    # 3. TRAIN HEADS ON FEATURES FROM PREDICTOR
                    if train_heads_on_predictor and train_heads:
                        if "image_head" in world_model.heads:
                            image_head = world_model.heads["image_head"]
                            head_pred_video_features, head_visual = fold_views_for_image_head(
                                image_head, pred_video_features, obs["visual"]
                            )
                            target_rgb_video = image_head.preprocess_rgb(head_visual[:, ::tubelet_size_enc])
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_image_losses = image_head.compute_loss(
                                    head_pred_video_features[:, :-1].detach(),
                                    target_rgb_video[:, 1:],
                                    global_step=epoch * ipe + itr,
                                )
                            rates["info/image_head/examples_seen"] += head_pred_video_features.shape[0] * (
                                head_pred_video_features.shape[1] - 1
                            )
                            predictor_image_losses = {k: v.mean() for k, v in predictor_image_losses.items()}
                            if train:
                                world_model.heads["image_head"].backward(predictor_image_losses["loss"])
                            predictor_image_losses = {
                                "predictor_image_" + k: v.item() for k, v in predictor_image_losses.items()
                            }
                            total_stats.update(predictor_image_losses)
                        if "state_head" in world_model.heads:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_state_losses = world_model.heads["state_head"].compute_loss(
                                    video_features, None, state[:, ::tubelet_size_enc]
                                )
                            rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                            predictor_state_losses = {k: v.mean() for k, v in predictor_state_losses.items()}
                            loss = predictor_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                            if train:
                                world_model.heads["state_head"].backward(predictor_state_losses["loss"])
                            predictor_state_losses = {
                                "predictor_state_" + k: v.item() for k, v in predictor_state_losses.items()
                            }
                            total_stats.update(predictor_state_losses)
                    # 4. TRAIN PREDICTOR ON FURTHER AUTOREGRESSIVE ROLLOUT
                    if rollout_steps > 1 and train and train_predictor:
                        if do_sequential_rollout:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                if train_rollout_prefixes == "random":
                                    prefixes = torch.randint(video_features.shape[1] - rollout_steps, size=(1,))
                                elif train_rollout_prefixes == "first":
                                    prefixes = [0]
                                elif train_rollout_prefixes == "all":
                                    prefixes = list(range(video_features.shape[1] - rollout_steps))
                                stats = defaultdict(list)
                                for t in prefixes:
                                    rollout_losses, total_rollout_loss, _, _ = world_model.rollout(
                                        video_features=video_features,
                                        pred_video_features=pred_video_features,
                                        proprio_features=proprio_features,
                                        pred_proprio_features=pred_proprio_features,
                                        action_features=action_features,
                                        action_noise=0.0,
                                        # we have `len(prefixes)` loss terms so weight them equally
                                        loss_weight=1.0 / len(prefixes),
                                        rollout_steps=rollout_steps - 1,
                                        rollout_stop_gradient=rollout_stop_gradient,
                                        ctxt_window=ctxt_window_train_rollout,
                                        mode="sequential",
                                        t=t,
                                    )
                                    total_transition_loss += total_rollout_loss
                                    for k in rollout_losses:
                                        stats[k].append(rollout_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        train_rollout_result[f"train_rollout/{k}/{j+2}"] = stats[k][j].item()
                        if do_parallel_rollout:
                            gt_prob = next(rollout_sampling_scheduler)
                            rates["info/transition_model/sampling_gt_prob"] = gt_prob
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                stats = defaultdict(list)
                                rollout_parallel_losses, total_rollout_parallel_loss, _, _ = world_model.rollout(
                                    video_features=video_features,
                                    proprio_features=proprio_features,
                                    action_features=action_features,
                                    pred_video_features=pred_video_features,
                                    pred_proprio_features=pred_proprio_features,
                                    action_noise=0.0,
                                    loss_weight=1.0,
                                    rollout_steps=rollout_steps - 1,
                                    rollout_stop_gradient=rollout_stop_gradient,
                                    ctxt_window=ctxt_window_train_rollout,
                                    mode="parallel",
                                    gt_prob=gt_prob,
                                    prepend_gt=prepend_gt_rollout_parallel,
                                )
                                total_transition_loss += total_rollout_parallel_loss
                                for k in rollout_parallel_losses:
                                    stats[k].append(rollout_parallel_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        parallel_rollout_result[f"train_rollout_parallel/{k}/{j+2}"] = stats[k][
                                            j
                                        ].item()
                    total_stats.update(train_rollout_result)
                    total_stats.update(parallel_rollout_result)

                    # Construct a clean losses dict that aggregates all relevant training losses
                    losses = {}
                    losses["predictor_loss"] = (
                        total_transition_loss.item()
                        if isinstance(total_transition_loss, torch.Tensor)
                        else total_transition_loss
                    )
                    losses["head_loss"] = (
                        total_head_loss.item() if isinstance(total_head_loss, torch.Tensor) else total_head_loss
                    )
                    losses["loss"] = losses["predictor_loss"] + losses["head_loss"]

                    grad_stats, optim_stats = {}, {}
                    # 5. OPTIMIZATION STEP
                    # so far, we only computed losses and ran .backward() so accumulate gradients on several objectives. Below is the
                    # only place where we perform an actual optimization step
                    if train:
                        if train_heads:
                            for name in world_model.heads.keys():
                                # If not train_heads, world_model.heads[name].model.module.decoder_embed.weight.grad should be None
                                grad_stats[name], optim_stats[name] = world_model.heads[name].optimization_step()
                        if train_predictor:
                            world_model.backward(total_transition_loss)
                            grad_stats["transition_model"], optim_stats["transition_model"] = (
                                world_model.optimization_step()
                            )
                        for key in list(grad_stats.keys()):
                            grad_stats[f"optim/{key}/grad_norm"] = (
                                grad_stats[key].global_norm if grad_stats[key] is not None else 0.0
                            )
                            del grad_stats[key]
                        for key in list(optim_stats.keys()):
                            optim_stats[f"optim/{key}/first_moment"] = (
                                optim_stats[key].get("exp_avg").avg if optim_stats[key] is not None else 0.0
                            )
                            optim_stats[f"optim/{key}/second_moment"] = (
                                optim_stats[key].get("exp_avg_sq").avg if optim_stats[key] is not None else 0.0
                            )
                            del optim_stats[key]
                    total_stats.update(grad_stats)
                    total_stats.update(optim_stats)
                    total_stats.update(dict(rates))
                    # 6. ONCE IN A WHILE DO LONG-ROLLOUT EVALUATION WITH DATASET ACTIONS OR RANDOM ACTIONS
                    eval_rollout_result = {}
                    # image_stats was initialized in Step 1 (so block 2's image_head
                    # visualization survives to here) -- not reset.
                    if not train:
                        world_model.eval()

                        @torch.no_grad
                        def val_rollout(
                            video_features,
                            action_features,
                            proprio_features,
                            pred_video_features,
                            pred_proprio_features,
                            gt_obs,
                            gt_state,
                            prefix_rollout_result=None,
                            val_rollout_steps=5,
                            ctxt_window=None,
                        ):
                            """
                            gt_obs is unused here: ground-truth comparison images are decoded
                            from video_features through image_head instead of read from
                            gt_obs["visual"] pixels, since that may hold precomputed embeddings
                            rather than RGB (model.visual_encoder.enc_type: precomputed). Kept as
                            a parameter for now since callers already pass it.
                            """
                            rollout_steps = min(val_rollout_steps, video_features.shape[1] - 1)
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                prefixes = range(video_features.shape[1] - rollout_steps)
                                val_rollout_result = {}

                                # Ground truth, reconstructed through the image_head decoder rather
                                # than read from gt_obs["visual"] pixels: when the world model
                                # consumes precomputed visual tokens (model.visual_encoder.enc_type:
                                # precomputed), obs["visual"] holds embeddings, not RGB, so there's
                                # no raw pixel tensor to fall back to. Decoded once here (b t v h w c,
                                # uint8) and reused below instead of once per run_rollout_and_decode
                                # call plus again for the rgb_v comparison image -- decoding through
                                # the depth-24 decoder isn't free.
                                gt_image_samples = (
                                    world_model.heads[image_head_name].decode(video_features[:, ::tubelet_size_enc])
                                    if image_head_name in world_model.heads
                                    else None
                                )

                                # Helper function to run rollout with different action noise levels and decode heads
                                def run_rollout_and_decode(action_noise, rollout_prefix):
                                    """Run rollout with given action noise and decode with image/state heads."""
                                    stats = defaultdict(list)
                                    last_prefix_t = None
                                    for t in prefixes:
                                        last_prefix_t = t
                                        rollout_losses, _, last_vid_feats, last_prop_feats = world_model.rollout(
                                            video_features=video_features,
                                            pred_video_features=pred_video_features,
                                            proprio_features=proprio_features,
                                            pred_proprio_features=pred_proprio_features,
                                            action_features=action_features,
                                            action_noise=action_noise,
                                            rollout_steps=rollout_steps,
                                            rollout_stop_gradient=True,
                                            debug=action_noise == 0.0,
                                            ctxt_window=ctxt_window,
                                            mode="sequential",
                                            t=t,
                                        )
                                        # last_vid_feats: [B T V H W D]
                                        for k in rollout_losses:
                                            stats[k].append(rollout_losses[k])

                                    image_samples = None
                                    if image_head_name in world_model.heads:
                                        image_samples = world_model.heads[image_head_name].decode(
                                            last_vid_feats[:, -rollout_steps - 1 :]
                                        )
                                        lpips_by_horizon = []
                                        for h in range(1, image_samples.shape[1]):
                                            # image_samples: b t v h w c. squeeze(2) only ever
                                            # dropped a size-1 view axis; with real multiview
                                            # (v>1) it's a no-op and permute(0,3,1,2) then sees
                                            # 5 dims instead of 4. Fold views into batch instead,
                                            # so this works for both single- and multi-view.
                                            pred = rearrange(image_samples[:, h], "b v h w c -> (b v) c h w")
                                            pred = pred.to(world_model.device, dtype=torch.float32) / 255.0

                                            # image_samples only ever holds the last prefix's
                                            # rollout tail (last_vid_feats is overwritten every
                                            # iteration of the loop above), so its local horizon
                                            # index h corresponds to absolute frame
                                            # last_prefix_t + h, not frame h of the full GT
                                            # sequence -- index gt_image_samples at that absolute
                                            # offset instead of at h directly.
                                            gt_frame = rearrange(
                                                gt_image_samples[:, last_prefix_t + h], "b v h w c -> (b v) c h w"
                                            )
                                            gt_frame = gt_frame.to(world_model.device, dtype=torch.float32) / 255.0

                                            # pred/gt_frame are already scaled to [0, 1]; lpips
                                            # defaults to normalize=False (expects [-1, 1]), so
                                            # without normalize=True this silently mis-scales
                                            # the metric instead of erroring.
                                            v = lpips(pred, gt_frame, normalize=True).mean().detach().cpu()
                                            lpips_by_horizon.append(v)
                                        if lpips_by_horizon:
                                            values = torch.stack(lpips_by_horizon)
                                            base = (
                                                f"{prefix_rollout_result}/{rollout_prefix}"
                                                if prefix_rollout_result
                                                else rollout_prefix
                                            )
                                            val_rollout_result[f"{base}/lpips_average"] = values.mean().item()

                                    if "state_head" in world_model.heads:
                                        target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                        state_loss = world_model.heads["state_head"].compute_loss(
                                            last_vid_feats[:, -rollout_steps - 1 :],
                                            None,
                                            target_states,
                                            reduce_mean=False,
                                        )
                                        for k, values in state_loss.items():
                                            values = values.mean(0)
                                            base = (
                                                f"{prefix_rollout_result}/{rollout_prefix}"
                                                if prefix_rollout_result
                                                else rollout_prefix
                                            )
                                            metric = _readable_name(f"decoded_state_{k}").removesuffix("_loss")
                                            val_rollout_result[f"{base}/{metric}_average"] = values.mean().item()

                                    # Aggregate rollout losses
                                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                    for k, values in stats.items():
                                        if k != "loss" and not k.startswith("proprio_"):
                                            weight_key = re.sub(r"^(visual|proprio)_", "", k) + "_weight"
                                            if cfgs_loss.get(weight_key, 0.0) == 0.0:
                                                continue
                                        base = (
                                            f"{prefix_rollout_result}/{rollout_prefix}"
                                            if prefix_rollout_result
                                            else rollout_prefix
                                        )
                                        metric = _readable_name(k).removesuffix("_loss")
                                        # compute_loss has already averaged batch and feature/token
                                        # axes. Finish by averaging rollout prefixes and horizons,
                                        # yielding exactly one scalar for every eval metric.
                                        val_rollout_result[f"{base}/{metric}_average"] = values.mean().item()

                                    return image_samples

                                eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.0, rollout_prefix="dataset_actions"
                                )

                                # Optionally decode position from ground truth visual features
                                if "state_head" in world_model.heads and data_traj_decode_gt:
                                    target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                    decode_gt_state_loss = world_model.heads["state_head"].compute_loss(
                                        video_features[:, -rollout_steps - 1 :], None, target_states, reduce_mean=False
                                    )
                                    for k, values in decode_gt_state_loss.items():
                                        values = values.mean(0)
                                        base = f"{prefix_rollout_result}/ground_truth_decode"
                                        metric = _readable_name(k).removesuffix("_loss")
                                        val_rollout_result[f"{base}/{metric}_average"] = values.mean().item()

                                noisy_eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.05, rollout_prefix="noisy_actions"
                                )

                                if "image_head" in world_model.heads:
                                    t = eval_image_samples.shape[1]
                                    b = min(4, video_features.shape[0])
                                    # gt_image_samples (decoded once above) is already b t v h w c,
                                    # uint8 -- no inverse_transform/rearrange needed, unlike the raw
                                    # gt_obs["visual"] pixels this used to read.
                                    rgb_v = gt_image_samples
                                    # Ground truth is always first/top in static comparisons.
                                    rgb = torch.stack([rgb_v[:, -t:], eval_image_samples], dim=2)[:b]
                                    rgb = (
                                        rearrange(
                                            rgb,
                                            "b t e v h w c -> (e b v h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    rgb = _label_comparison(rgb, "GROUND TRUTH", "IMAGINED", "height")
                                    rgb_noised = torch.stack([rgb_v[:, -t:], noisy_eval_image_samples], dim=2)[:b]
                                    rgb_noised = (
                                        rearrange(
                                            rgb_noised,
                                            "b t e v h w c -> (e b v h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    rgb_noised = _label_comparison(
                                        rgb_noised, "GROUND TRUTH", "IMAGINED (NOISY ACTIONS)", "height"
                                    )
                                    animation = torch.stack([rgb_v[:, -t:], eval_image_samples], dim=2)[:b]
                                    animation = (
                                        rearrange(
                                            animation,
                                            "b t e v h w c -> t c (b v h) (e w)",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    animation = _label_comparison(
                                        animation, "GROUND TRUTH", "IMAGINED", "width"
                                    )
                                    animation_noised = torch.stack(
                                        [rgb_v[:, -t:], noisy_eval_image_samples], dim=2
                                    )[:b]
                                    animation_noised = (
                                        rearrange(
                                            animation_noised,
                                            "b t e v h w c -> t c (b v h) (e w)",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    animation_noised = _label_comparison(
                                        animation_noised, "GROUND TRUTH", "IMAGINED (NOISY ACTIONS)", "width"
                                    )
                                else:
                                    rgb, rgb_noised, animation, animation_noised = None, None, None, None
                            return rgb, rgb_noised, animation, animation_noised, val_rollout_result

                        if do_data_traj_rollout_eval:
                            rgb, rgb_noised, animation, animation_noised, eval_rollout_result = val_rollout(
                                video_features,
                                action_features,
                                proprio_features,
                                pred_video_features,
                                pred_proprio_features,
                                gt_obs=obs,
                                prefix_rollout_result="data_traj",
                                val_rollout_steps=cfgs_data_traj_rollout_eval.get("data_traj_eval_rollout_steps", 3),
                                ctxt_window=cfgs_data_traj_rollout_eval.get("data_traj_eval_ctxt_window", None),
                                gt_state=state,
                            )
                        if do_energy_landscape_eval:
                            gt_visual = inverse_transform(obs["visual"].cpu())
                            gt_visual = (255.0 * gt_visual).clip(0.0, 255.0).to(torch.uint8)
                            gt_proprio = preprocessor.denormalize_proprios(obs["proprio"].cpu())
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                energy_landscape = world_model.compute_energy_landscape(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                    gt_visual,
                                    gt_proprio,
                                    rollout_steps=cfgs_energy_landscape_eval.get("energy_landscape_rollout_steps", 1),
                                    ctxt_window=cfgs_energy_landscape_eval.get("energy_landscape_ctxt_window", None),
                                    proprio_dim=traj_dataset.proprio_dim,
                                    action_dim=traj_dataset.action_dim,
                                    actions_per_vid_feat=actions_per_vid_feat,
                                    dataset_path=dataset_paths[0],
                                    preprocessor=preprocessor,
                                )
                            image_stats.update(
                                {
                                    "data_traj/energy_landscape": energy_landscape,
                                }
                            )
                        if "image_head" in world_model.heads:
                            if do_data_traj_rollout_eval:
                                image_stats.update(
                                    {
                                        "data_traj/ground_truth_then_imagined": wandb.Image(rgb),
                                        "data_traj/ground_truth_then_imagined_noisy_actions": wandb.Image(rgb_noised),
                                        # Each video has ground truth on the left and imagination on the right.
                                        "data_traj/ground_truth_vs_imagined": wandb.Video(
                                            animation,
                                            caption="Ground truth (left) | Imagined (right). Camera rows per episode: " + ", ".join(cfgs_camera.get("camera_views", [])),
                                            fps=6,
                                            format="gif",
                                        ),
                                        "data_traj/ground_truth_vs_imagined_noisy_actions": wandb.Video(
                                            animation_noised,
                                            caption="Ground truth (left) | Imagined with noisy actions (right). Camera rows per episode: " + ", ".join(cfgs_camera.get("camera_views", [])),
                                            fps=6,
                                            format="gif",
                                        ),
                                    }
                                )
                        world_model.train()
                    total_stats.update(eval_rollout_result)
                    return (
                        float(losses["loss"]),
                        losses,
                        optim_stats,
                        total_stats,
                        image_stats,
                    )

                # In train mode, image_stats is empty
                if not light_eval_only_mode:
                    obs, action, state, reward, masks_enc, masks_pred = get_batch()
                    (loss, losses, optim_stats, total_stats, image_stats), gpu_etime_ms = gpu_timer(
                        lambda: step_model(obs, action, state, reward, train=True)
                    )
                    iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
                    loss_meter.update(loss)
                    gpu_time_meter.update(gpu_etime_ms)
                    wall_time_meter.update(iter_elapsed_time_ms)
                    if train_csv_logger is None:  # Initialize the logger once
                        train_csv_logger = create_csv_logger(losses, total_stats, train=True)
                else:
                    losses = {}
                    total_stats = {}

                if itr % light_eval_freq == light_eval_freq - 1 and val_loader_iters is not None:
                    # image_stats overrides the empty image_stats from the train step at same itr
                    image_stats, eval_losses, eval_total_stats = {}, {}, {}
                    # Then, use non-distributed validation data for visualization on rank 0
                    if val_viz_rank0_loader and rank == 0:
                        viz_obs, viz_action, viz_state, viz_reward, _, _ = get_viz_batch()
                        if viz_obs is not None:
                            with torch.no_grad():
                                (_, _, _, _, viz_img_stats), _ = gpu_timer(
                                    lambda: step_model(viz_obs, viz_action, viz_state, viz_reward, train=False)
                                )
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in viz_img_stats.items()}
                                    )
                                else:
                                    image_stats.update(viz_img_stats)
                    # First, use distributed validation data for metrics
                    for idx in range(len(val_loader_iters)):
                        val_name = val_loader_names[idx]
                        obs, action, state, reward, masks_enc, masks_pred = get_batch(train=False, idx=idx)
                        with torch.no_grad():
                            (_, val_losses, _, val_total_stats, val_img_stats), gpu_etime_ms = gpu_timer(
                                lambda: step_model(
                                    obs, action, state, reward, train=False, head_key=val_loader_head_keys[idx]
                                )
                            )
                            if not light_eval_only_mode:
                                # Optimizer/action diagnostics and zero-valued training losses only
                                # add dashboard noise during eval. Rollout quality is the useful output.
                                rollout_stats = {k: v for k, v in val_total_stats.items() if k.startswith("data_traj/")}
                                eval_total_stats.update({f"eval/{val_name}/{k}": v for k, v in rollout_stats.items()})
                            # Only use distributed batch images if we don't have visualization images
                            if not val_viz_rank0_loader:
                                prefixed_stats = {f"eval/{val_name}/{k}": v for k, v in val_img_stats.items()}
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in prefixed_stats.items()}
                                    )
                                else:
                                    image_stats.update(prefixed_stats)
                    if eval_csv_logger is None:  # Initialize the logger once
                        eval_csv_logger = create_csv_logger(eval_losses, eval_total_stats, train=False)
                else:
                    eval_losses = {}
                    eval_total_stats = {}

                # -- Logging
                def log_stats():
                    trainer.log(epoch, itr, losses, total_stats, eval_losses, eval_total_stats, image_stats)
                    if not light_eval_only_mode:
                        log_values = [epoch + 1, itr, loss, gpu_etime_ms, iter_elapsed_time_ms]
                        for key in train_csv_logger_columns[5:]:
                            if key in losses:
                                value = losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in total_stats:
                                value = total_stats[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            else:
                                log_values.append(0.0)
                        train_csv_logger.log(*log_values)
                        if rank == 0:
                            epoch_pbar.set_postfix(
                                loss=f"{loss_meter.avg:.3f}",
                                mem=f"{torch.cuda.max_memory_allocated() / 1024.0**3:.1f}GiB",
                            )
                        if np.isnan(loss) or np.isinf(loss):
                            logger.info(
                                "[%d, %5d] "
                                "[mem: %.2e] "
                                "[gpu: %.1f ms]"
                                "[wall: %.1f ms]"
                                % (
                                    epoch + 1,
                                    itr,
                                    torch.cuda.max_memory_allocated() / 1024.0**2,
                                    gpu_time_meter.avg,
                                    wall_time_meter.avg,
                                )
                            )
                    if itr % light_eval_freq == light_eval_freq - 1:
                        log_values = [epoch + 1, itr]
                        for key in eval_csv_logger_columns[2:]:
                            if key in eval_losses:
                                value = eval_losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in eval_total_stats:
                                log_values.append(eval_total_stats[key])
                            else:
                                log_values.append(0.0)
                        eval_csv_logger.log(*log_values)

                log_stats()
                if not light_eval_only_mode:
                    assert not np.isnan(loss), "loss is nan"
            logger.info("avg. loss %.3f" % loss_meter.avg)

            # -- Save Last
            if not light_eval_only_mode:
                if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
                    if rank == 0:
                        save_checkpoint(epoch + 1, latest_path)
                        if save_every_freq > 0 and epoch % save_every_freq == 0:
                            save_every_file = pref_tag + f"e{epoch}.{latest_format}"
                            save_every_path = os.path.join(checkpoint_folder, save_every_file)
                            save_checkpoint(epoch + 1, save_every_path)

            # -- Launch Planning Eval
            if not light_eval_only_mode:
                if (epoch % eval_freq == 0) or epoch == (num_epochs - 1):
                    if save_every_freq > 0:
                        checkpoint = (
                            pref_tag + f"latest.{latest_format}"
                            if epoch == (num_epochs - 1)
                            else pref_tag + f"e{epoch}.{latest_format}"
                        )
                    else:
                        checkpoint = pref_tag + f"latest.{latest_format}"
                    launch_planning_evals(
                        rank,
                        epoch + 1,
                        folder,
                        checkpoint,
                        cfgs_plan_evals,
                        cfgs_model,
                        cfgs_data,
                        cfgs_data_aug,
                        "",
                        world_model=world_model,
                        dset=val_traj_dataset,
                        preprocessor=preprocessor,
                        checkpoint_folder=checkpoint_folder,
                    )
                    world_model.train()
    elif unroll_decode_eval_only_mode:
        logger.info("Launching unroll-decode evals only mode")
        checkpoint = pref_tag + f"latest.{latest_format}"
        launch_unroll_decode_eval(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_unroll_decode_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
        )
    else:
        logger.info("Skipping training loop due to plan_only_eval_mode being enabled.")
        checkpoint = pref_tag + f"latest.{latest_format}"
        launch_planning_evals(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_plan_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
            "-plan-only",
            world_model=world_model,
            dset=val_traj_dataset,
            preprocessor=preprocessor,
            checkpoint_folder=checkpoint_folder,
        )


def launch_planning_evals(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_plan_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
    tag_suffix,
    world_model=None,
    dset=None,
    preprocessor=None,
    checkpoint_folder=None,
):
    """
    Launch planning evaluations for the current training checkpoint.

    This function generates complete eval configs by merging training model/data settings
    with eval config templates, then either submits distributed eval jobs via sbatch
    or runs them locally (if separate=False).

    The eval config generation flow:
    1. Load eval config templates from cfgs_plan_evals["eval_cfg_paths"]
       (typically located in configs/online_plan_evals/)
    2. Call build_plan_eval_args() to merge training configs (model, data, data_aug)
       with these templates
    3. Either dump configs for debugging or submit/run eval jobs

    Config options in cfgs_plan_evals:
    - dump_eval_configs (bool): If True, dump generated configs to disk and exit early
      without launching evals. The dump directory is automatically derived from
      eval_cfg_paths (e.g., "configs/online_plan_evals/mz/..." -> "configs/dump_online_evals/mz/").
      Output filenames are derived from the template basenames.
    - separate (bool): If True (default), submit eval jobs via sbatch. If False, run
      evals on rank 0 of the current training job.

    To generate eval configs without running training (e.g., for an already-trained model):
    1. Set meta.plan_only_eval_mode: true in your training config
    2. Set evals.dump_eval_configs: true in your training config
    3. Run: python -m app.main --fname <your_config.yaml> --debug
    4. Configs will be saved to configs/dump_online_evals/<env>/ (derived from eval_cfg_paths)

    Args:
        rank: Process rank in distributed training
        epoch: Current training epoch
        folder: Output folder path for the training run
        checkpoint: Checkpoint filename to evaluate
        cfgs_plan_evals: Evaluation configuration dict from training config
        cfgs_model: Model configuration from training config
        cfgs_data: Data configuration from training config
        cfgs_data_aug: Data augmentation configuration from training config
        tag_suffix: Suffix to append to evaluation tags
        world_model: Optional loaded world model (for non-separate eval mode)
        dset: Optional validation dataset (for non-separate eval mode)
        preprocessor: Optional data preprocessor (for non-separate eval mode)
    """
    eval_cfg_paths = cfgs_plan_evals.get("eval_cfg_paths", None)
    eval_nodes = cfgs_plan_evals.get("nodes", None)
    eval_episodes = cfgs_plan_evals.get("eval_episodes", None)
    eval_low_pri = cfgs_plan_evals.get("low_pri", True)
    separate = cfgs_plan_evals.get("separate", True)
    override_cfgs_data = cfgs_plan_evals.get("override_cfgs_data", True)
    override_datasets = cfgs_plan_evals.get("override_datasets", True)
    # task_specification
    evals_obs = cfgs_plan_evals.get("obs", None)
    # planner
    evals_alpha = cfgs_plan_evals.get("alpha", None)
    max_episode_steps = cfgs_plan_evals.get("max_episode_steps", None)
    num_act_stepped = cfgs_plan_evals.get("num_act_stepped", None)
    horizon = cfgs_plan_evals.get("horizon", None)
    evals_decode = cfgs_plan_evals.get("decode", None)
    sum_all_diffs = cfgs_plan_evals.get("sum_all_diffs", None)
    goal_H = cfgs_plan_evals.get("goal_H", None)
    num_elites = cfgs_plan_evals.get("num_elites", None)
    if eval_cfg_paths is not None:
        eval_nodes, eval_tasks_per_node, args_eval, eval_cpus_per_task = build_plan_eval_args(
            app_name="vjepa_wm",
            folder=folder,
            checkpoint=checkpoint,
            eval_cfg_paths=eval_cfg_paths,
            cfgs_model=cfgs_model,
            cfgs_data=cfgs_data,
            cfgs_data_aug=cfgs_data_aug,
            override_cfgs_data=override_cfgs_data,
            override_datasets=override_datasets,
            tag=f"epoch-{epoch}{tag_suffix}",
            evals_decode=evals_decode,
            sum_all_diffs=sum_all_diffs,
            evals_obs=evals_obs,
            evals_alpha=evals_alpha,
            eval_nodes=eval_nodes,
            eval_episodes=eval_episodes,
            max_episode_steps=max_episode_steps,
            num_act_stepped=num_act_stepped,
            horizon=horizon,
            goal_H=goal_H,
            num_elites=num_elites,
            wrapper_kwargs=cfgs_plan_evals.get("wrapper_kwargs", {}),
            checkpoint_folder=checkpoint_folder,
        )

        # Dump eval configs if in dump_eval_configs mode (useful for generating configs without training)
        dump_eval_configs = cfgs_plan_evals.get("dump_eval_configs", False)
        if dump_eval_configs:
            if rank == 0:
                # Deduce dump directory from eval_cfg_paths
                # e.g., "configs/online_plan_evals/mz/ng/..." -> "configs/dump_online_evals/mz/"
                first_template = eval_cfg_paths[0] if eval_cfg_paths else None
                if first_template and "online_plan_evals" in first_template:
                    # Extract environment name from path (e.g., "mz", "pt", "wall", "droid")
                    parts = first_template.split("online_plan_evals/")
                    if len(parts) > 1:
                        env_part = parts[1].split("/")[0]  # Get first directory after online_plan_evals/
                        dump_dir = f"configs/dump_online_evals/{env_part}"
                    else:
                        dump_dir = "configs/dump_online_evals"
                else:
                    dump_dir = "configs/dump_online_evals"
                os.makedirs(dump_dir, exist_ok=True)
                dumped_paths = []
                for i, cfg in enumerate(args_eval):
                    # Derive output filename from the config's tag field
                    # e.g., "online_gc_zeroshot/wall_L2_ng_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96/epoch-50-plan-only"
                    # -> "wall_L2_ng_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml"
                    tag = cfg.get("tag", None)
                    if tag:
                        tag_parts = tag.split("/")
                        if len(tag_parts) >= 2:
                            output_name = tag_parts[-2] + ".yaml"
                        else:
                            output_name = tag_parts[0] + ".yaml"
                    else:
                        output_name = f"eval_config_{i}.yaml"
                    output_path = os.path.join(dump_dir, output_name)
                    dump_yaml(cfg, output_path)
                    dumped_paths.append(output_path)
                logger.info(f"Dumped {len(args_eval)} eval configs:\n" + "\n".join(f"  - {p}" for p in dumped_paths))
            import sys

            sys.exit(0)

        for i, cfg in enumerate(args_eval):
            args_eval[i] = convert_to_dict_recursive(args_eval[i])

        if separate:
            if rank == 0:
                account, partition, qos = slurm_account_partition_and_qos(low_pri=eval_low_pri)
                logger.info(f"Launching online evals with {account=}, {partition=}, {qos=}")
                with submitit.helpers.clean_env():
                    launch_evals(
                        args_for_evals=args_eval,
                        nodes=eval_nodes,
                        tasks_per_node=eval_tasks_per_node,
                        submitit_folder=os.path.join(folder, "submitit-evals"),
                        account=account,
                        partition=partition,
                        qos=qos,
                        cpus_per_task=eval_cpus_per_task,
                        delay_seconds=5,
                        timeout=120,  # to schedule faster, could be insufficient if using old GPUs making eval slow
                    )
                logger.info(f"Launched online evals from templates {eval_cfg_paths}")
        else:
            from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
            from evals.simu_env_planning.eval import main_distributed_episodes_eval as gc_main_dist

            world_model.eval()
            for i, cfg in tqdm(enumerate(args_eval)):
                eval_tag = cfg.get("tag", None)
                pretrain_folder = cfg.get("folder", None)
                folder = os.path.join(pretrain_folder, "simu_env_planning/")
                if eval_tag is not None:
                    folder = os.path.join(folder, eval_tag)
                cfg["frameskip"] = cfg["model_kwargs"]["data"]["custom"]["frameskip"]
                cfg["work_dir"] = folder
                model = EncPredWM(
                    world_model,
                    action_dim=world_model.action_dim,
                    preprocessor=preprocessor,
                    ctxt_window=cfg["model_kwargs"]["wrapper_kwargs"]["ctxt_window"],
                )
                gc_main_dist(cfg, model=model, dset=dset, preprocessor=preprocessor, rank=rank)


def launch_unroll_decode_eval(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_unroll_decode_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
):
    """
    Launch unroll decode evaluations for counterfactual decoding of unrolled predictions.

    This evaluation hardcodes custom actions (e.g., open/close gripper + move up) to generate
    counterfactual decodings, allowing visual comparison of different action scenarios.

    Config options in cfgs_unroll_decode_evals:
    - dump_eval_configs (bool): If True, dump generated configs to disk and exit early
      without launching evals (useful for generating configs without training)
    - specific_video (bool): If True, use a specific video file instead of dataset samples
    - specific_video_path (str): Path to specific video file (npz format)
    - play_in_reverse (bool): If True, reverse the video sequence
    - obs (str): Observation type - "rgb" or "rgb_state"
    - save_decoding_only (bool): If True, only save decoded predictions (not ground truth comparison)
    - repeat_hardcode_act (int): Number of times to repeat the hardcoded action sequence
    - wrapper_kwargs (dict): Model wrapper configuration (same as evals.wrapper_kwargs)
        - ctxt_window (int): Context window size for the model wrapper
        - proprio_mode (str): Proprioception mode (e.g., "compute_new_pose")

    Args:
        rank: Process rank in distributed training
        epoch: Current training epoch
        folder: Output folder path for the training run
        checkpoint: Checkpoint filename to evaluate
        cfgs_unroll_decode_evals: Unroll decode evaluation configuration dict
        cfgs_model: Model configuration from training config
        cfgs_data: Data configuration from training config
        cfgs_data_aug: Data augmentation configuration from training config
    """
    # Build evaluation arguments
    args_eval = build_unroll_decode_eval_args(
        app_name="vjepa_wm",
        folder=folder,
        checkpoint=checkpoint,
        cfgs_model=cfgs_model,
        cfgs_data=cfgs_data,
        cfgs_data_aug=cfgs_data_aug,
        cfgs_unroll_decode_evals=cfgs_unroll_decode_evals,
        tag=f"epoch-{epoch}",
    )

    # Dump eval configs if in dump_eval_configs mode (useful for generating configs without training)
    dump_eval_configs = cfgs_unroll_decode_evals.get("dump_eval_configs", False)
    if dump_eval_configs:
        if rank == 0:
            dump_dir = "configs/dump_online_evals/vjepa_wm/unroll_decode"
            os.makedirs(dump_dir, exist_ok=True)
            for i, cfg in enumerate(args_eval):
                yaml_path = os.path.join(dump_dir, f"unroll_decode_{i}.yaml")
                dump_yaml(cfg, yaml_path)
            logger.info(f"Dumped {len(args_eval)} unroll_decode eval configs to {dump_dir}")
        # All ranks exit early after dumping configs (skip launching evals)
        return

    for i, cfg in enumerate(args_eval):
        args_eval[i] = convert_to_dict_recursive(args_eval[i])

    # Run eval directly on rank 0
    from evals.unroll_decode.eval import main as unroll_decode_main

    if rank == 0:
        for cfg in args_eval:
            unroll_decode_main(cfg)
