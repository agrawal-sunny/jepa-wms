# UR lift JEPA-WM implementation review

Reviewed 2026-09-07. Scope: the UR lift environment and shared collectors/wrapper, Isaac NPZ adapter and trajectory splitting, active DINO-WM predictor and embeddings, losses, optimizer, rollout evaluation, logging/checkpoints, current configuration, and the latest completed run `logs/isaac_lift_env_ur/run-20260904-175217` (W&B `jyadvjd8`). This is a source/data/checkpoint review, not a simulator validation or a hyperparameter sweep. Training defaults were not changed during this review.

## Findings to address before tuning

### 1. High: the learned proprioception target has nearly collapsed

[video_wm.py](app/vjepa_wm/video_wm.py:399) supervises predicted proprioception against the trainable proprio encoder's output. The one-step target is not detached, and [utils.py](app/vjepa_wm/utils.py:913) optimizes that encoder. Reducing the variation in the target is an easy way to reduce the loss without improving physical pose prediction.

This is supported by the final checkpoint, not just a theoretical concern. Its 20-dimensional linear proprio embedding has weight norm **0.02587**, bias norm **2.0105**, and mean per-channel standard deviation **0.000717** when evaluated on all current UR poses. Its mean output vector has norm **2.0121**: the representation is almost constant. The recorded final proprio L2 of approximately **2.8e-6** is therefore not evidence of accurate Cartesian prediction. A direct autograd probe also confirmed gradients flowing into the target.

Use a fixed, well-scaled pose representation as the prediction target, or add supervised prediction/reconstruction of physical pose to keep the learned representation informative. Merely detaching the target does not by itself guarantee prevention of collapse. Log translation error in meters and rotation error in degrees, plus embedding variance. Do not try to solve this by increasing the current proprio loss weight.

### 2. High: timeout transitions contain the next episode's reset observation

The [random collector](../takeoff/collect_random_demos.py:309) saves the observation returned by `env.step()` even when `done` is true. Isaac Lab [resets before returning observations](../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:216), and the [wrapper](../dreamer_wrapper.py:116) sets `is_first=0` unconditionally on these steps. The [adapter](app/plan_common/datasets/isaac_npz_dset.py:157) includes those frames as ordinary dynamics targets.

Of the **600 current random trajectories**, **449** have 301 observations (initial observation plus the 300-step time limit); **361** of those have a final EEF position jump greater than **10 cm**. This matches the reset path in the source. Success/failure collection for this lift configuration normally ends from observation flags before an environment reset; its current trajectories showed no such final jumps above 10 cm.

Capture terminal observations before reset for future collection. For existing timeout episodes, exclude the invalid final observation and aligned transition when loading, preserving synchronized fields. Do not remove every terminal observation indiscriminately: genuine success/failure terminal frames are useful.

### 3. High: LPIPS uses inconsistent pixel ranges and mismatched timestamps

[train.py](app/vjepa_wm/train.py:1203) decodes only the final rollout prefix. For the active 8-frame validation clip and 5-step forecast, its decoded sequence corresponds to frames **2–7**, but LPIPS compares predicted frames **3–7** against ground-truth frames **1–5**. The latent rollout loss uses separately aligned targets and is not affected by this timestamp bug.

Predicted pixels are divided by 255 into `[0,1]`, while the current ground truth is normalized to `[-1,1]`. Both LPIPS inputs must use the same expected range; the [official LPIPS documentation](https://github.com/richzhang/PerceptualSimilarity#aii-python-code) specifies `[-1,1]`. Align the ground-truth suffix to the decoded prefix, inverse-transform ground truth, and scale both identically. Report that metric explicitly as final-prefix LPIPS, or compute it across every prefix. Include ground-truth-embedding decoder reconstruction as a baseline to separate decoder limitations from dynamics errors.

### 4. Medium: training CSV headers do not describe the stored values

[create_csv_logger](app/vjepa_wm/train.py:329) registers `epoch, itr` and sorted metrics, while [log_stats](app/vjepa_wm/train.py:1480) writes `epoch, itr, loss, gpu_time, iter_time` and those metrics. The CSV writer zips values with formats, shifting column names and dropping the final three values. For example, the third column is labeled `act_max` but contains loss. W&B uses a separate dictionary and is not affected by this specific defect.

Define one consistent column schema and write exactly those fields. Existing training losses were recovered from the third positional column for this review; do not parse historical training CSVs by their current headers. Evaluation CSVs do not have this defect.

### 5. Medium: the “noisy actions” comparison replaces action embeddings

[video_wm.py](app/vjepa_wm/video_wm.py:610) replaces encoded actions with zero-mean Gaussian noise rather than adding noise to the recorded raw actions. This is neither a 0.05 perturbation of the original command nor necessarily an encoding of any valid command. It also makes comparisons change as the action encoder's scale changes.

For an action-sensitivity comparison, perturb or shuffle raw actions before encoding, handling the binary gripper command separately. Add a persistence baseline and a shuffled-action baseline, and compare them on exactly the same held-out clips.

## Dataset balance and validation quality

The current folder contains **1,005 episodes**: 600 random, 205 success, 200 failure. Reconstructing the adapter's actual recursive-glob path order and split with seed 234 gives:

| Demo type | Training episodes | Training 4-frame clips | Share of training clips |
|---|---:|---:|---:|
| Random | 540 | 138,569 | 87.6% |
| Success | 185 | 11,402 | 7.2% |
| Failure | 179 | 8,270 | 5.2% |

All random trajectories hold the gripper open. Uniform sampling of overlapping clips therefore substantially emphasizes open-gripper random motion over grasping and failure dynamics. Start by testing **50% random / 25% success / 25% failure**, sampling trajectories within each group to avoid long episodes dominating. This needs sampler support: `datasets_weights` is not applied by the current `isaac_npz` branch of [init_data](app/plan_common/datasets/utils.py:89).

The current filtered validation split contains 20 success and 21 failure trajectories, totaling **1,974 eight-frame clips**. Each light evaluation consumes only one batch of four clips, then advances the iterator. The completed run logged 39 such evaluations per epoch. An epoch mean therefore samples different validation clips from the previous epoch rather than evaluating a fixed full set. Windows from the same episode are also correlated.

Use a fixed held-out set for checkpoint comparisons, report front/wrist and success/failure separately, and aggregate per trajectory. Compute cheap latent metrics over that set once per epoch; keep decoded videos to a small fixed subset. Preserve a separate test split for the final chosen settings. Save the best checkpoint based on the stable validation metric: currently each epoch overwrites `jepa-latest`, and `save_every_freq=-1` disables retained epoch checkpoints.

## What the convergence estimate actually means

The saved run configuration used **front camera only** and did **not** specify today's `val_demo_types` filter. Today's two-camera, task-only validation setup is a different experiment. The 20-epoch default remains a reasonable initial budget, but the old run cannot establish convergence for the changed setup.

| Epochs | Mean training objective | Mean validation latent rollout loss |
|---|---:|---:|
| 1 | 0.182846 | 0.503738 |
| 6–10 | 0.126570 | 0.386683 |
| 11–20 | 0.119322 | 0.390538 |
| 21–50 | 0.110483 | 0.392052 |

These are distinct objectives: training uses teacher-forced one-step predictions and an extra factor of 1/2 with `rollout_steps=1`, while validation measures autoregressive multi-step predictions. The numbers support a plateau in the old validation metric around epoch 10, not a claim that all learned dynamics converged correctly.

## Prioritized hyperparameter experiments

The following values are proposed experiments, not measured optima. Establish a corrected baseline first and change one family of settings at a time.

| Priority | Setting | Current | First experiment |
|---|---|---|---|
| 1 | Demo sampling | 87.6% random clips | 50% random / 25% success / 25% failure |
| 2 | Training forecast horizon | `rollout_steps=1` | Test 2, then 3; for 3 use `num_pred=3`, `num_hist=3`, `num_frames_pred=6` |
| 3 | Learning-rate schedule | Constant `5e-4`, no warmup | `start_lr=1e-5`, `ref_lr=3e-4`, `final_lr=3e-5`, `warmup=0.25` epoch; compare peaks `1e-4`, `3e-4`, `5e-4` |
| 4 | Weight decay | `1e-7` to `1e-6` | Start with constant `1e-4`; compare `1e-5`, `1e-4`, `1e-3` |
| 5 | Batch size | 8 | Test 16 if memory permits; record optimizer steps and examples seen because epochs alone hide the changed update count |
| 6 | Predictor capacity | 6 layers, 16 heads | Consider 3 layers only after fixing the objective/data issues; increasing capacity is not supported by these curves |

At `sim.dt=0.01` and decimation 5, collection runs at **20 Hz**. One training step predicts 50 ms ahead; validation forecasts 250 ms. Three context observations span just 100 ms. Multi-step training is therefore a more direct experiment than adding more epochs. Keep `frameskip=1` initially, frozen DINOv2, and the existing decoder-compatible image normalization. The stored DINOv3 arrays are not interchangeable with this DINOv2 encoder's features.

## Configuration traps and secondary issues

- `normalize_action=true` currently also normalizes proprioception and fits statistics on **all trajectories before splitting** ([adapter](app/plan_common/datasets/isaac_npz_dset.py:112)). Fit statistics only on training trajectories before testing this option. Keep binary gripper semantics explicit.
- `prop_mlp=true` currently fails with the active 20-dimensional proprio embedding: [utils.py](app/vjepa_wm/utils.py:819) constructs an MLP expecting 384 inputs. A direct forward probe reproduced the matrix-shape error.
- Setting `freeze_encoder=false` does not unfreeze DINO: [init_video_model](app/vjepa_wm/utils.py:690) unconditionally sets its parameters' `requires_grad=False`. Encoder fine-tuning would also require explicit target and checkpoint handling.
- The `dino_wm` predictor hardcodes dropout 0.1 and MLP width 2048 in [utils.py](app/vjepa_wm/utils.py:718). `attn.local_window_time` does not configure this predictor branch. Its causal mask and rollout context are controlled elsewhere.
- `sampling_scheduler` is unused while `do_parallel_rollout=false`. `eval_freq` controls planning evaluation, which has no configured evaluation paths here; it does not control the light rollout metrics. For the NPZ adapter, clip lengths come from `num_hist + num_pred` and `num_frames_val`, not `dataset_fpcs` or `fps`.
- [WM construction](app/vjepa_wm/train.py:735) reads `frameskip` and `action_skip` from the outer data dictionary instead of `data.custom`. Both happen to be 1 now; verify propagation before changing temporal subsampling.
- Camera selection uses a dataset-owned `RandomState` copied into workers. Seed it per worker before relying on independent camera sampling. Validation does correctly pair both configured cameras at the same timestamps.
- The recursive-glob branch of `_trajectory_paths` does not sort its results. The same seed can produce a different train/validation assignment if filesystem enumeration order changes. Sort paths and save an explicit split manifest; changing ordering now also changes the old split, so preserve its manifest when making that fix.
- `_load_episode` decompresses the full NPZ camera member to obtain a slice, then loads the chosen camera again for shape checking. Its “only requested camera frames” docstring overstates the optimization. Consider a cache or clip-friendly storage after profiling; there is no need to tune the network around this I/O limitation.
- `compute_loss` computes cosine similarities without a denominator clamp even when cosine weight is zero. A zero-feature probe produced `NaN` in an L2-only loss because `0 * NaN` remains `NaN`. Guard norms and avoid evaluating unused losses. This is a reproducible edge case, not an observed NaN in the completed run.
- The lift success threshold is center height **0.20 m**, compared with spawn center **0.191 m**, and has no hold duration ([task_terms.py](../lift_env/mdp/task_terms.py:30)). That is about 9 mm above spawn and can end successful collection early. Failure requires an outer block tilt above **1 radian** (~57 degrees). These are task-definition choices retained from the legacy environment; confirm the intended lift/hold behavior before changing them and collecting a new dataset. Reward weights and curricula do not enter this world-model objective (`with_reward=false`, no reward head).

## Verification performed

- Read the active pipeline and reconstructed the current trajectory split without decompressing images.
- Inspected all current UR episodes' compact arrays and the final checkpoint's action/proprio embeddings.
- Recomputed training and validation aggregates from the completed run's CSVs, accounting for the training header defect.
- Existing Isaac adapter tests: **5 passed** (action shift, schema, synchronized lengths, validation filtering, paired cameras).
- Existing DINO-WM ViTPredictor tests: **2 passed**.
- Direct probes confirmed target-side gradients, the `prop_mlp` shape failure, and the zero-feature loss NaN.
- No GPU training sweep or simulator rollout was launched; none of the proposed hyperparameter values has been validated as optimal.
