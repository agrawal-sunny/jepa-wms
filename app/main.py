# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import multiprocessing as mp
import pprint
from datetime import datetime
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.yaml_utils import expand_env_vars

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1"],
    help=(
        "physical CUDA devices to use (default: cuda:0 cuda:1). "
        "Pass one device, for example '--devices cuda:1', for single-GPU training without --debug."
    ),
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If specified, will not spawn child processes. "
    "The training code runs in the launcher process, which makes it easier to \
    debug with checkpointing.",
)
parser.add_argument(
    "--resume-run",
    type=str,
    metavar="RUN_DIR",
    help="resume an existing run directory (the default is a normal independent run)",
)
parser.add_argument(
    "--resume-checkpoint",
    type=str,
    metavar="FILENAME",
    default=None,
    help="checkpoint filename inside --resume-run to load instead of the default "
    "'<write_tag>-latest.<latest_format>' (e.g. 'jepa-e15.pth.tar'); requires --resume-run. "
    "The wandb run tied to that run directory (its wandb_run_id.txt) still resumes "
    "the same way regardless of which checkpoint file is chosen.",
)


def process_main(
    rank,
    fname,
    world_size,
    devices,
    preserve_visible_devices=False,
    run_instance_id=None,
    resume_run_dir=None,
    resume_checkpoint=None,
):
    import os

    requested_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not (preserve_visible_devices and requested_visible_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "CUDA device mapping: physical CUDA_VISIBLE_DEVICES=%s; this process uses logical cuda:0",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<all GPUs visible>"),
    )
    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        # Expand environment variables in folder as early as possible
        if "folder" in params:
            params["folder"] = expand_env_vars(params["folder"], _path="folder")
        # Timestamped subdirectories isolate checkpoints, but they should not
        # change the human-facing experiment name.
        default_run_name = Path(params["folder"]).name
        params.setdefault("logging", {}).setdefault("wandb", {}).setdefault("run_name", default_run_name)
        if resume_run_dir is not None:
            resume_path = Path(resume_run_dir).expanduser()
            if not resume_path.is_absolute():
                cwd_candidate = Path.cwd() / resume_path
                resume_path = cwd_candidate if cwd_candidate.exists() else Path(params["folder"]) / resume_path
            write_tag = params.get("logging", {}).get("write_tag", "jepa")
            latest_format = params.get("logging", {}).get("latest_format", "pth.tar")
            checkpoint_name = resume_checkpoint or (
                f"{write_tag}-latest.{latest_format}" if write_tag else f"latest.{latest_format}"
            )
            checkpoint_path = resume_path / checkpoint_name
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Cannot resume: checkpoint not found at {checkpoint_path}")
            params["folder"] = str(resume_path.resolve())
            params.setdefault("meta", {})["load_checkpoint"] = True
            if resume_checkpoint:
                # Explicit filename, honored regardless of finetune/"-latest" state
                # (see meta.read_checkpoint handling in app/vjepa_wm/train.py). The
                # wandb run tied to this folder (wandb_run_id.txt) resumes the same
                # way either way -- it doesn't depend on which checkpoint file this is.
                params["meta"]["read_checkpoint"] = resume_checkpoint
            logger.info("Resume requested: output folder=%s, checkpoint=%s", params["folder"], checkpoint_name)
        elif run_instance_id is not None:
            params["folder"] = str(Path(params["folder"]) / run_instance_id)
            params.setdefault("meta", {})["load_checkpoint"] = False
            params["meta"]["read_checkpoint"] = None
            logger.info("Run output folder: %s", params["folder"])
        logger.info("✅ Config loaded")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config
    app_main(params["app"], args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.resume_checkpoint and not args.resume_run:
        parser.error("--resume-checkpoint requires --resume-run")
    run_instance_id = None if args.resume_run else datetime.now().strftime("run-%Y%m%d-%H%M%S")
    if args.debug:
        # Respect an explicit CUDA_VISIBLE_DEVICES setting in debug mode. Previously
        # this path silently replaced e.g. CUDA_VISIBLE_DEVICES=1 with GPU 0.
        process_main(
            rank=0,
            fname=args.fname,
            world_size=1,
            devices=[args.devices[0]],
            preserve_visible_devices=True,
            run_instance_id=run_instance_id,
            resume_run_dir=args.resume_run,
            resume_checkpoint=args.resume_checkpoint,
        )
    else:
        num_gpus = len(args.devices)
        mp.set_start_method("spawn")
        for rank in range(num_gpus):
            mp.Process(
                target=process_main,
                args=(rank, args.fname, num_gpus, args.devices),
                kwargs={
                    "run_instance_id": run_instance_id,
                    "resume_run_dir": args.resume_run,
                    "resume_checkpoint": args.resume_checkpoint,
                },
            ).start()
