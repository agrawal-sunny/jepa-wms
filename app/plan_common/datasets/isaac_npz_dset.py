"""Isaac Lab trajectory adapter for JEPA-WM.

The project collector stores ``observation[t]`` together with a dummy action at
index zero; ``action[t]`` produced ``observation[t]`` from
``observation[t - 1]``.  JEPA-WM expects actions aligned as
``action[t]: observation[t] -> observation[t + 1]``.  This adapter performs
that one-step shift and pads the final action.
"""

from __future__ import annotations

import glob
from copy import copy
from pathlib import Path
from typing import Iterable, Literal, Sequence
from zipfile import ZipFile

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from .traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset, get_train_val_sliced


def _npz_shape(path: Path, key: str) -> tuple[int, ...]:
    """Read an array shape from its NPY header without decompressing its data."""
    with ZipFile(path) as archive, archive.open(f"{key}.npy") as member:
        version = np.lib.format.read_magic(member)
        shape, _, _ = np.lib.format._read_array_header(member, version)
    return shape


def _trajectory_paths(values: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value).expanduser()
        matches = sorted(candidate.rglob("*.npz")) if candidate.is_dir() else [Path(p) for p in glob.glob(str(candidate), recursive=True)]
        paths.extend(path.resolve() for path in matches if path.is_file() and path.suffix == ".npz")
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError(f"No Isaac Lab .npz trajectories matched: {list(values)}")
    return paths


class IsaacNPZDataset(TrajDataset):
    """Read Takeoff and Lift-Env episodes saved by the shared collector."""

    def __init__(
        self,
        data_paths: Sequence[str | Path],
        transform=None,
        camera_views: Sequence[str] | str = ("front_cam",),
        normalize_action: bool = False,
        with_reward: bool = False,
        dset_fraction: float = 1.0,
        seed: int = 0,
        action_key: Literal["action", "latent_action_lam"] = "action",
        embedding_suffix: str | None = None,
    ):
        # When set, reads precomputed "{camera}_{embedding_suffix}" token
        # arrays (written by preprocess_dino.py) instead of raw
        # "{camera}" RGB frames -- see video_wm.py's enc_type == "precomputed".
        self.embedding_suffix = embedding_suffix
        if action_key not in ("action", "latent_action_lam"):
            raise ValueError(f"Unsupported action_key: {action_key!r}")
        self.action_key = action_key
        self.paths = _trajectory_paths(data_paths)
        if action_key == "latent_action_lam":
            # LAM latent-action width varies with the checkpoint (e.g. the
            # legacy 4x512 LAM vs. a narrower 32-dim one) -- read it from the
            # cache instead of assuming 2048, so a differently-sized LAM just
            # works once precompute_latent_actions.py has regenerated the
            # cache for it.
            first_shape = _npz_shape(self.paths[0], action_key)
            if len(first_shape) != 2:
                raise ValueError(f"{self.paths[0]}:{action_key} must be T,D, got {first_shape}")
            self.action_dim = first_shape[1]
        else:
            self.action_dim = 7
        if not 0 < dset_fraction <= 1:
            raise ValueError(f"dset_fraction must be in (0, 1], got {dset_fraction}")
        self.paths = self.paths[: max(1, int(len(self.paths) * dset_fraction))]
        self.samples = self.paths  # used by TrajSubset for diagnostics
        self.transform = transform
        self.camera_views = (camera_views,) if isinstance(camera_views, str) else tuple(camera_views)
        if not self.camera_views:
            raise ValueError("At least one camera view is required")
        self.all_camera_views = False
        self.with_reward = with_reward
        self.rng = np.random.RandomState(seed)
        self.seq_lengths: list[int] = []

        action_chunks: list[torch.Tensor] = []
        proprio_chunks: list[torch.Tensor] = []
        for path in tqdm(self.paths, desc="Scanning Isaac Lab trajectories"):
            with np.load(path) as episode:
                visual_keys = [self.visual_key(camera) for camera in self.camera_views]
                required = [*visual_keys, self.action_key, "eef_pos", "eef_quat"]
                if with_reward:
                    required.append("reward")
                missing = [key for key in required if key not in episode]
                if missing:
                    raise ValueError(f"{path} is missing required fields: {missing}")
                compact_keys = [self.action_key, "eef_pos", "eef_quat"] + (["reward"] if with_reward else [])
                lengths = {key: len(episode[key]) for key in compact_keys}
                for camera, key in zip(self.camera_views, visual_keys):
                    shape = _npz_shape(path, key)
                    if self.embedding_suffix is not None:
                        if len(shape) != 3:
                            raise ValueError(f"{path}:{key} must be T,N,D precomputed tokens, got {shape}")
                    elif len(shape) != 4 or shape[-1] not in (3, 4):
                        raise ValueError(f"{path}:{key} must be T,H,W,C RGB(A), got {shape}")
                    lengths[key] = shape[0]
                if len(set(lengths.values())) != 1:
                    raise ValueError(f"{path} has unsynchronized fields: {lengths}")
                length = next(iter(lengths.values()))
                if length < 2:
                    raise ValueError(f"{path} needs at least two observations, got {length}")
                actions = np.asarray(episode[self.action_key], dtype=np.float32)
                positions = np.asarray(episode["eef_pos"], dtype=np.float32)
                quaternions = np.asarray(episode["eef_quat"], dtype=np.float32)
                if actions.shape != (length, self.action_dim):
                    raise ValueError(f"{path}:{self.action_key} must have shape (T, {self.action_dim}), got {actions.shape}")
                if positions.shape != (length, 3) or quaternions.shape != (length, 4):
                    raise ValueError(f"{path}: expected eef_pos (T,3) and eef_quat (T,4)")
                if not all(np.isfinite(array).all() for array in (actions, positions, quaternions)):
                    raise ValueError(f"{path} contains non-finite actions or proprioception")
                self.seq_lengths.append(length)
                action_chunks.append(torch.from_numpy(actions[1:].copy()))
                proprio_chunks.append(torch.from_numpy(np.concatenate((positions, quaternions), axis=-1)))

        self.proprio_dim = 7
        self.state_dim = 7
        all_actions = torch.cat(action_chunks)
        all_proprios = torch.cat(proprio_chunks)
        if normalize_action:
            self.action_mean = all_actions.mean(0)
            self.action_std = all_actions.std(0).clamp_min(1e-6)
            self.proprio_mean = all_proprios.mean(0)
            self.proprio_std = all_proprios.std(0).clamp_min(1e-6)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
        self.state_mean = self.proprio_mean.clone()
        self.state_std = self.proprio_std.clone()

    def __len__(self):
        return len(self.paths)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def visual_key(self, camera: str) -> str:
        """NPZ member name for one camera: raw frames, or its precomputed embedding."""
        return camera if self.embedding_suffix is None else f"{camera}_{self.embedding_suffix}"

    def __getitem__(self, idx, **kwargs):
        return (*self._load_episode(idx),)

    def _load_episode(self, idx, frame_slice=slice(None), action_slice=slice(None)):
        """Load one episode or clip, decompressing only the requested camera frames."""
        with np.load(self.paths[idx]) as episode:
            camera = self.camera_views[self.rng.randint(len(self.camera_views))]
            cameras = self.camera_views if self.all_camera_views else (camera,)
            visuals = []
            for view in cameras:
                key = self.visual_key(view)
                if self.embedding_suffix is not None:
                    # Precomputed patch tokens (t, n, d): already the encoder's
                    # output, so no pixel decode, resize, or normalize transform.
                    tokens = episode[key][frame_slice]
                    visual = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
                else:
                    frames = episode[key][frame_slice, ..., :3]
                    visual = torch.from_numpy(np.asarray(frames, dtype=np.float32))
                    visual = rearrange(visual, "t h w c -> t c h w")
                    if visual.max() > 1:
                        visual = visual / 255.0
                    if self.transform:
                        visual = self.transform(visual)
                visuals.append(visual)
            visual = torch.stack(visuals, dim=1) if self.all_camera_views else visuals[0]
            key = self.visual_key(camera)
            all_frames = episode[key]
            if self.embedding_suffix is not None:
                if all_frames.ndim != 3:
                    raise ValueError(f"{self.paths[idx]}:{key} must be T,N,D precomputed tokens, got {all_frames.shape}")
            elif all_frames.ndim != 4 or all_frames.shape[-1] not in (3, 4):
                raise ValueError(f"{self.paths[idx]}:{key} must be T,H,W,C RGB(A), got {all_frames.shape}")
            if len(all_frames) != self.seq_lengths[idx]:
                raise ValueError(
                    f"{self.paths[idx]}:{key} has {len(all_frames)} frames; expected {self.seq_lengths[idx]}"
                )
            raw_actions = torch.from_numpy(np.asarray(episode[self.action_key], dtype=np.float32))
            # Shift collector convention to JEPA-WM convention and pad the
            # action after the final observation.
            actions = torch.cat((raw_actions[1:], torch.zeros_like(raw_actions[:1])), dim=0)
            actions[:-1] = (actions[:-1] - self.action_mean) / self.action_std
            actions = actions[action_slice]
            proprio = torch.from_numpy(
                np.concatenate((episode["eef_pos"], episode["eef_quat"]), axis=-1).astype(np.float32)
            )
            proprio = (proprio - self.proprio_mean) / self.proprio_std
            proprio = proprio[frame_slice]
            reward = (
                torch.from_numpy(np.asarray(episode["reward"], dtype=np.float32))[frame_slice]
                if self.with_reward
                else None
            )
        obs = {"visual": visual, "proprio": proprio}
        return obs, actions, proprio.clone(), reward, {"path": str(self.paths[idx]), "camera": camera}

    def get_slice(self, idx, start, end, frameskip, action_skip, process_actions):
        """Load a training clip without decoding and transforming the full trajectory."""
        obs, actions, state, reward, _ = self._load_episode(
            idx,
            frame_slice=slice(start, end, frameskip),
            action_slice=slice(start, end, action_skip),
        )
        num_frames = len(range(start, end, frameskip))
        if reward is None:
            reward = torch.zeros(num_frames, dtype=torch.float32)
        if frameskip < action_skip:
            action_frames = num_frames * frameskip // action_skip
            actions = rearrange(actions, "(n f) d -> n (f d)", n=action_frames)
        elif process_actions == "concat":
            actions = rearrange(actions, "(n f) d -> n (f d)", n=num_frames)
        elif process_actions == "sum":
            actions = rearrange(actions, "(n f) d -> n f d", n=num_frames).sum(dim=1)
        else:
            raise ValueError(f"Unknown process_actions mode: {process_actions!r}")
        return obs, actions, state, reward


def _filter_by_demo_type(subset, dataset, demo_types, num_frames, frameskip, action_skip, random_seed, process_actions, label):
    """Restrict a TrajSubset to episodes stored under the given collector folders.

    Shared by the val_demo_types filter below and filter_train_by_demo_types:
    filters by collection folder (dataset.paths[i].parent.name), not terminal
    flags, since random-policy episodes can also terminate in success/failure.
    """
    demo_types = {demo_types} if isinstance(demo_types, str) else set(demo_types)
    if not demo_types or not demo_types <= {"success", "failure", "random"}:
        raise ValueError(f"Invalid demo_types: {demo_types!r}")
    filtered = TrajSubset(dataset, [i for i in subset.indices if dataset.paths[i].parent.name in demo_types])
    filtered_slices = TrajSlicerDataset(
        filtered,
        num_frames,
        frameskip,
        action_skip,
        generator=torch.Generator().manual_seed(random_seed),
        process_actions=process_actions,
    )
    if not len(filtered_slices):
        raise ValueError(f"No {label} clips match demo_types={demo_types!r}")
    return filtered, filtered_slices


def load_isaac_npz_slice_train_val(
    transform,
    data_paths,
    normalize_action=False,
    split_ratio=0.9,
    num_hist=3,
    num_pred=1,
    num_frames_val=None,
    frameskip=1,
    action_skip=1,
    traj_subset=True,
    random_seed=42,
    with_reward=False,
    process_actions="concat",
    camera_views=("front_cam",),
    dset_fraction=1.0,
    val_demo_types=None,
    # Applies val_demo_types to the *train* split instead of (not in addition
    # to) the valid split -- e.g. for a training-data rollout-visualization
    # loader that should still only show success/failure episodes. See
    # train.py's val_datasets_N "split: train" handling.
    filter_train_by_demo_types=False,
    action_key="action",
    embedding_suffix=None,
):
    dataset = IsaacNPZDataset(
        data_paths=data_paths,
        transform=transform,
        camera_views=camera_views,
        normalize_action=normalize_action,
        with_reward=with_reward,
        dset_fraction=dset_fraction,
        seed=random_seed,
        action_key=action_key,
        embedding_suffix=embedding_suffix,
    )
    # Multiview: feed every configured camera together (encoded separately,
    # concatenated as extra per-frame tokens -- see VideoWM.encode_obs) rather
    # than randomly sampling one view per clip. Applied to the shared dataset
    # object so both the train and validation splits inherit it.
    dataset.all_camera_views = len(dataset.camera_views) > 1
    train, valid, train_slices, valid_slices = get_train_val_sliced(
        traj_dataset=dataset,
        train_fraction=split_ratio,
        random_seed=random_seed,
        num_frames=num_hist + num_pred,
        num_frames_val=num_frames_val,
        frameskip=frameskip,
        action_skip=action_skip,
        traj_subset=traj_subset,
        process_actions=process_actions,
    )
    if val_demo_types is not None and not filter_train_by_demo_types:
        valid, valid_slices = _filter_by_demo_type(
            valid, dataset, val_demo_types, num_frames_val or num_hist + num_pred,
            frameskip, action_skip, random_seed, process_actions, "held-out validation",
        )
    if filter_train_by_demo_types:
        if val_demo_types is None:
            raise ValueError("filter_train_by_demo_types requires val_demo_types to be set")
        train, train_slices = _filter_by_demo_type(
            train, dataset, val_demo_types, num_hist + num_pred,
            frameskip, action_skip, random_seed, process_actions, "training",
        )
    # Keep paired views together in each validation clip, sharing timestamps/actions.
    validation_dataset = copy(dataset)
    validation_dataset.all_camera_views = len(dataset.camera_views) > 1
    valid.dataset = validation_dataset
    return {"train": train_slices, "valid": valid_slices}, {"train": train, "valid": valid}


def flatten_camera_batch(obs, action, state, reward):
    """Evaluate paired cameras as independent samples with the single-view model."""
    if obs["visual"].ndim != 6:
        return obs, action, state, reward
    views = obs["visual"].shape[2]
    obs = {
        key: rearrange(value, "b t v c h w -> (b v) t c h w")
        if key == "visual" else value.repeat_interleave(views, dim=0)
        for key, value in obs.items()
    }
    return (obs, *(value.repeat_interleave(views, dim=0) for value in (action, state, reward)))
