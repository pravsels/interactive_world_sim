# train_stage1 - front cam WAN h5 baseline

## Mode
- run_type: replication
- objective: verify stage1 training runs end-to-end on Isambard with WAN-format h5 data.

## Config
- script: `slurm/iws_train_stage1_slurm.sh`
- config: `configurations/isambard_train.yaml`
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_wan224.h5`
- key settings:
  - dataset: `arx5_h5_dataset`
  - obs_keys: `['camera_1_color']` (front only)
  - camera mapping: `camera_0=wrist`, `camera_1=front`
  - action key: `actions_delta` (dim=7)
  - training stage: `1` (autoencoder)
  - wandb mode: `offline`
  - checkpoint every: `2000` train steps
  - walltime: `1-00:00:00`

## Job
- job_id: `2732782` (first submission)
- submitted: `2026-03-11 12:31 UTC`
- node: `nid010979`
- submit command: `sbatch slurm/iws_train_stage1_slurm.sh`

## Job (resumed)
- job_id: `2732865`
- submitted: `2026-03-11 12:32 UTC`
- resumed from: `same run, fixed config path resolution (no checkpoint load)`
- node: `nid010985`

## Job (debug rerun)
- job_id: `2733054`
- submitted: `2026-03-11 12:38 UTC`
- resumed from: `same run, fixed GLIBC/OpenCV lib path in container`

## Job (debug rerun)
- job_id: `2733207`
- submitted: `2026-03-11 12:41 UTC`
- resumed from: `same run, validating runtime after previous fix`
- node: `nid010949`

## Job (debug rerun)
- job_id: `2733422`
- submitted: `2026-03-11 12:48 UTC`
- resumed from: `same run, testing numcodecs runtime fallback`
- node: `nid010981`

## Job (debug rerun)
- job_id: `2733506`
- submitted: `2026-03-11 12:50 UTC`
- resumed from: `same run, with python extension bind mount`
- node: `nid010988`

## Job (debug rerun)
- job_id: `2733619`
- submitted: `2026-03-11 12:53 UTC`
- resumed from: `same run, switched to srun + apptainer module pattern`
- node: `nid010958`

## Job (fresh run)
- job_id: `2733840`
- submitted: `2026-03-11 13:03 UTC`
- resumed from: `clean submission after canceling previous queued/running jobs`
- node: `nid010635`

## Job (debug rerun)
- job_id: `2786521`
- submitted: `2026-03-12 12:58 UTC`
- resumed from: `diagnostic run with num_workers=0`
- node: `nid010577`

## Job (debug rerun)
- job_id: `2786567`
- submitted: `2026-03-12 13:04 UTC`
- resumed from: `num_workers=0 after DataLoader prefetch fix`
- node: `nid010541`

## Job (debug rerun)
- job_id: `2786672`
- submitted: `2026-03-12 13:15 UTC`
- resumed from: `num_workers=0 after GH200 attention backend fix`
- node: `nid010365`

## Job (debug rerun)
- job_id: `2786694`
- submitted: `2026-03-12 13:22 UTC`
- resumed from: `thread-capped debug run (OMP/MKL/OPENBLAS/NUMEXPR=1, num_workers=0)`
- node: `nid010739`

## Job (debug rerun)
- job_id: `2786723`
- submitted: `2026-03-12 13:28 UTC`
- resumed from: `step-0 instrumentation run (IWS_DEBUG_STEP_TRACE=1, thread caps, num_workers=0)`
- node: `nid010621`

## Job (debug rerun)
- job_id: `2786774`
- submitted: `2026-03-12 13:36 UTC`
- resumed from: `hook instrumentation run (IWS_DEBUG_STEP_TRACE=1, IWS_DEBUG_HOOK_TRACE=1, thread caps, num_workers=0)`
- node: `nid011090`

## Job (debug rerun)
- job_id: `2786935`
- submitted: `2026-03-12 13:41 UTC`
- resumed from: `CUDA_LAUNCH_BLOCKING=1 run with step+hook traces, thread caps, num_workers=0`
- node: `nid010714`

## Job (debug rerun)
- job_id: `2787050`
- submitted: `2026-03-12 13:53 UTC`
- resumed from: `forced pure math SDPA backend (IWS_FORCE_SDPA_MATH=1) with CUDA_LAUNCH_BLOCKING=1 + step/hook traces + thread caps + num_workers=0`
- node: `nid011115`

## Job (debug rerun)
- job_id: `2788937`
- submitted: `2026-03-12 14:42 UTC`
- resumed from: `single-GPU isolation rerun: Slurm mem changed to 128G and exclusive node allocation removed; trainer strategy now keyed to configured num_devices*num_nodes`
- node: `nid010811`

## Job (debug rerun)
- job_id: `2788966`
- submitted: `2026-03-12 14:46 UTC`
- resumed from: `same single-GPU isolation debug setup, re-submitted after commit/push and fresh pull on Isambard`
- node: `nid010879`

## Job (debug rerun)
- job_id: `2789217`
- submitted: `2026-03-12 14:45 UTC`
- resumed from: `strict dummy-loss probe (IWS_DEBUG_DUMMY_LOSS=1) with no model forward; same single-GPU isolation + hook tracing + thread caps + num_workers=0`
- node: `nid010507`

## Job (debug rerun)
- job_id: `2789543`
- submitted: `2026-03-12 15:02 UTC`
- resumed from: `existing pipeline run with Lightning Trainer bypass enabled (IWS_DEBUG_MANUAL_TORCH_LOOP=1, 3 manual steps, same model+dataloader)`
- node: `nid010545`

## Job (debug rerun)
- job_id: `2789711`
- submitted: `2026-03-12 15:14 UTC`
- resumed from: `LD-path-only A/B test: skip legacy LD patch (IWS_SKIP_LD_PATCH=1) while keeping prior manual-loop debug settings`
- node: `nid010257`

## Job (debug rerun)
- job_id: `2789735`
- submitted: `2026-03-12 15:24 UTC`
- resumed from: `latent_safety-style manual torch loop refinement: disable Lightning hook calls (IWS_DEBUG_MANUAL_CALL_HOOKS=0) while keeping manual loop + same data/model`
- node: `nid010570`

## Status
- 2026-03-11 - prepared run log before first submission.
- 2026-03-11 12:31 UTC - job `2732782` failed in 6s, exit code `1:0`.
- failure reason: `Training config yaml not found: /var/spool/slurmd/job.../../configurations/isambard_train.yaml`
- fix applied: default `TRAIN_CONFIG_YAML` now resolves from `REPO_DIR/configurations/isambard_train.yaml` instead of script directory.
- 2026-03-11 12:32 UTC - job `2732865` failed, exit code `1:0`.
- failure reason: `ImportError: ... GLIBC_2.38 not found` from OpenCV dependency chain.
- fix applied: set `LD_LIBRARY_PATH` inside `apptainer exec` runtime before launching training.
- 2026-03-11 12:38 UTC - job `2733054` failed, exit code `1:0` (follow-up debug run).
- 2026-03-11 12:41 UTC - job `2733207` failed in 18s, exit code `1:0`.
- failure reason: `ImportError: cannot import name 'cbuffer_sizes' from 'numcodecs.blosc'` (zarr/numcodecs API mismatch).
- fix applied: slurm script now prepends scratch `PYTHON_EXT_DIR` via `PYTHONPATH` and installs `numcodecs==0.11.0` there only when needed.
- 2026-03-11 12:48 UTC - job `2733422` failed in 65s, exit code `2:0`.
- failure reason: `OSError: [Errno 30] Read-only file system` while pip writing to `${PYTHON_EXT_DIR}` from inside container.
- fix applied: bind-mount `${PYTHON_EXT_DIR}` into container and create it before runtime install.
- 2026-03-11 12:50 UTC - job `2733506` failed in 72s, exit code `1:0`.
- failure reason: `RuntimeError: Found no NVIDIA driver on your system` when constructing `algorithm.dynamics`.
- fix applied: align with working `latent_safety` pattern by loading `brics/apptainer-multi-node` and launching container through `srun --gpus=1`.
- 2026-03-11 12:53 UTC - job `2733619` failed in 22s, exit code `1:0` with same NVIDIA driver error.
- root-cause hypothesis: `LD_LIBRARY_PATH` override removed Apptainer/NVIDIA injected libs path.
- fix applied: preserve existing `LD_LIBRARY_PATH` and prepend required CUDA/system paths.
- 2026-03-11 13:01 UTC - committed/pushed `configurations/isambard_train.yaml` update: `experiment.training.batch_size=16` (`stage1 batch 16`).
- 2026-03-11 13:03 UTC - canceled old jobs and submitted clean run as `2733840`.
- 2026-03-11 13:06 UTC - verification on Isambard for `2733840`: job is `RUNNING` on `nid010635` (`squeue`/`sacct`), and live GPU sample via `srun --jobid=2733840 --overlap nvidia-smi` reports `59342 MiB / 97871 MiB` used.
- 2026-03-12 12:29 UTC - follow-up check on Isambard for `2733840`: still `RUNNING` on `nid010635` (`squeue`/`sacct`), elapsed `23:25:19` / limit `1-00:00:00`, exit code currently `0:0`.
- 2026-03-12 12:31 UTC - inspected `slurm-2733840.out/.err` and run output dir (`/scratch/u6cr/pravsels.u6cr/interactive_world_sim/outputs/2026-03-11/13-04-13`): logs contain only startup/model-init lines, no train/val loss lines, no checkpoints/metrics files, and W&B offline files have metadata only (no history/summary). Live GPU sample shows `0%` util with `59340 MiB` allocated, suggesting the job is likely stalled before first logged step.
- 2026-03-12 12:36 UTC - deep process check on compute node (`srun --jobid=2733840 --overlap`): main `python main.py` process is alive and waiting (`futex_wait_queue`), with 8 child `python main.py` worker processes in poll waits and large H5 read bytes already accumulated. This pattern is consistent with a dataloader/input pipeline stall (workers alive but no batches reaching trainer), so no loss logs/checkpoints are produced.
- 2026-03-12 12:44-12:47 UTC - continuous GPU telemetry sample on job `2733840` (multiple 10s samples): GPU util remained `0%` and memory util `0%` throughout while memory stayed pinned at `59339 MiB / 97871 MiB` and process power draw stayed ~`138.4-138.7 W`. Confirms allocated GPU memory without active training compute.
- 2026-03-12 12:58 UTC - canceled stuck job `2733840` (`scancel 2733840`). Final state: `CANCELLED`.
- 2026-03-12 12:58 UTC - submitted diagnostic job `2786521` with overrides `experiment.training.data.num_workers=0 experiment.validation.data.num_workers=0`.
- 2026-03-12 12:59 UTC - job `2786521` failed quickly with `ValueError`: DataLoader `prefetch_factor` cannot be set when `num_workers=0`.
- 2026-03-12 13:01 UTC - fix committed/pushed: DataLoader now sets `prefetch_factor=1` only when `num_workers > 0`.
- 2026-03-12 13:04 UTC - submitted debug rerun `2786567` with `num_workers=0` after fix.
- 2026-03-12 13:06 UTC - job `2786567` is `RUNNING` on `nid010541`, but still no train/val loss lines or checkpoints yet; one-step GPU sample still shows `0%` util with `~59338 MiB` allocated.
- 2026-03-12 13:07 UTC - updated root-cause hypothesis: stall is likely not worker-count only; attention backend selection may be wrong on GH200 because code treats all `sm>=8.0` + `.minor==0` GPUs as A100 and forces flash-attention backend.
- 2026-03-12 13:15 UTC - patched and pushed attention backend selection: now only true A100 (`name contains A100` and `sm_80`) uses flash-attention-only path; GH200 uses math/efficient SDPA backends.
- 2026-03-12 13:15 UTC - canceled `2786567`, pulled latest on Isambard, and submitted rerun `2786672` with `num_workers=0`.
- 2026-03-12 13:16 UTC - `2786672` confirms patched backend path in logs (`Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda`) and reaches `training_step` entry (`Top 10 memory diff from start to step 0`), but still no step/loss prints or checkpoints.
- 2026-03-12 13:17 UTC - spot check for `2786672`: still `RUNNING`, GPU sample `util=0%`, memory `59338 MiB / 97871 MiB`; stall remains unresolved.
- 2026-03-12 13:20 UTC - live attach debugging on `2786672`: gdb attach could not read usable symbols from containerized python binary (`Input/output error`), but `strace` sample shows main and worker threads mostly blocked in `futex(...WAIT...)` and repeated `ppoll(...)=0 (Timeout)` loops (no active compute/data syscalls), confirming the process is synchronization-idle rather than progressing through training steps.
- 2026-03-12 13:22 UTC - canceled `2786672` and submitted `2786694` with thread caps (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) plus `num_workers=0`.
- 2026-03-12 13:23-13:24 UTC - monitored `2786694`: behavior unchanged. Job reaches startup/model summary and `Top 10 memory diff from start to step 0` line, but no train/val loss lines or checkpoints. GPU sample remains idle (`util=0%`, memory `59338 MiB / 97871 MiB`).
- 2026-03-12 13:28 UTC - submitted `2786723` with step-0 markers enabled (`IWS_DEBUG_STEP_TRACE=1`) on top of thread-capped + `num_workers=0` settings.
- 2026-03-12 13:29-13:30 UTC - marker output for `2786723` reaches all stage-1 checkpoints through `stage1_after_log` on batch 0 (`after_tracemalloc` -> `before/after_normalize` -> `before/after_encoder` -> `before/after_noise_levels` -> `before/after_pred_s` -> `before/after_pred_u` -> `before/after_log`).
- 2026-03-12 13:30 UTC - despite full batch-0 `training_step` completion, run still emits no loss/progress/checkpoint artifacts afterward. Updated hypothesis: stall is likely in Lightning post-`training_step` path (backward/optimizer/synchronization), not inside dataset load or stage-1 forward/loss code.
- 2026-03-12 13:36 UTC - submitted `2786774` with hook markers enabled (`IWS_DEBUG_HOOK_TRACE=1`) on top of step markers + thread caps + `num_workers=0`.
- 2026-03-12 13:37-13:38 UTC - `2786774` prints hook markers through `on_before_zero_grad` and `on_before_backward`, but never prints `on_after_backward` or optimizer-step markers.
- 2026-03-12 13:38 UTC - refined root-cause localization: stall occurs during/inside backward pass (after `on_before_backward`, before backward completes), not in data loading, forward, loss computation, or pre-backward Lightning plumbing.
- 2026-03-12 13:41 UTC - canceled `2786774` and submitted `2786935` with `CUDA_LAUNCH_BLOCKING=1` + `HYDRA_FULL_ERROR=1` while keeping step/hook traces and debug caps.
- 2026-03-12 13:42-13:43 UTC - `2786935` shows the exact same marker cutoff (`on_before_backward` is last hook reached; no `on_after_backward`), and no traceback surfaced in stderr even with launch blocking.
- 2026-03-12 13:43 UTC - launch-blocking result indicates a hard/stuck backward kernel path (hang without thrown CUDA exception), likely in autograd/backward compute rather than asynchronous error propagation.
- 2026-03-12 13:53 UTC - canceled `2786935`, patched attention backend with runtime toggle `IWS_FORCE_SDPA_MATH=1`, and submitted `2787050` to force math-only SDPA during the same backward-path diagnostic setup.
- 2026-03-12 13:57 UTC - `2787050` is currently `PENDING` (no node assigned yet), so no slurm output exists yet for this run.
- 2026-03-12 14:09 UTC - `2787050` started on `nid011115` and reproduces the same marker cutoff: full batch-0 `training_step` markers through `stage1_after_log`, then hooks reach `on_before_zero_grad` and `on_before_backward` but do not reach `on_after_backward`.
- 2026-03-12 14:09 UTC - no traceback in stderr while running; forced math SDPA did not immediately unblock backward in this diagnostic.
- 2026-03-12 14:42 UTC - canceled `2787050`, applied single-GPU isolation changes (`#SBATCH --mem=128G`, removed `#SBATCH --exclusive`) and updated trainer strategy selection to use config (`num_devices*num_nodes > 1`) rather than visible node GPU count.
- 2026-03-12 14:42 UTC - submitted rerun `2788937` with same debug flags (`IWS_DEBUG_STEP_TRACE=1`, `IWS_DEBUG_HOOK_TRACE=1`, `CUDA_LAUNCH_BLOCKING=1`, thread caps, `num_workers=0`, `IWS_FORCE_SDPA_MATH=1`) to isolate allocator/strategy effects.
- 2026-03-12 14:45 UTC - workflow correction: committed/pushed single-GPU changes to GitHub (`bb686c7`), then pulled latest `main` on Isambard before running again.
- 2026-03-12 14:46 UTC - canceled `2788937` and submitted synced rerun `2788966` (same debug flags) from pulled commit state; job is running on `nid010879`.
- 2026-03-12 14:49 UTC - committed/pushed dummy-loss debug toggle (`2f2056c`), pulled latest on Isambard, canceled `2788966`, and submitted strict dummy-loss run `2789217` (`IWS_DEBUG_DUMMY_LOSS=1`).
- 2026-03-12 14:47-14:48 UTC - `2789217` confirms dummy-loss path is active (`skipping model forward and returning dummy scalar loss`) yet still stalls after `on_before_backward` with no `on_after_backward`.
- 2026-03-12 14:48 UTC - key conclusion from strict probe: the hang reproduces even without model forward/backward graph, so root cause is likely outside model autograd graph (Lightning/runtime/CUDA environment path).
- 2026-03-12 14:49 UTC - canceled `2789217` after capturing markers; final Slurm state `CANCELLED`.
- 2026-03-12 14:52 UTC - ran an isolated in-container CUDA smoke probe (same SIF + `--nv`, outside Lightning training loop): `torch 2.10.0+cu126`, CUDA available on `NVIDIA GH200 120GB`, cuDNN `91002`, and both basic matmul backward + SDPA backward completed successfully.
- 2026-03-12 14:52 UTC - conclusion from smoke probe: container CUDA libraries are functional for core PyTorch backward paths; the hang is likely specific to training-runtime integration (Lightning loop/optimizer plugin/runtime interactions), not a blanket CUDA-lib failure.
- 2026-03-12 15:01 UTC - committed/pushed Lightning bypass patch in existing training pipeline (`84d990b`): when `IWS_DEBUG_MANUAL_TORCH_LOOP=1`, `training()` skips `Trainer.fit` and runs a short manual torch optimizer loop over the same model/dataloader.
- 2026-03-12 15:02 UTC - pulled latest on Isambard and submitted `2789543` with manual-loop flag enabled (`IWS_DEBUG_MANUAL_TORCH_LOOP=1`, `IWS_DEBUG_MANUAL_STEPS=3`, plus prior step/hook/thread debug flags).
- 2026-03-12 15:03-15:04 UTC - `2789543` reproduces the same cutoff in manual mode: full step-0 markers through `stage1_after_log`, `on_before_backward` printed, but no `on_after_backward` and no manual step completion print.
- 2026-03-12 15:04 UTC - key conclusion: hang persists even when bypassing Lightning `Trainer.fit`; issue is deeper than high-level Lightning loop plumbing.
- 2026-03-12 15:04 UTC - canceled `2789543` after capture; final Slurm state `CANCELLED`.
- 2026-03-12 15:13 UTC - committed/pushed Slurm toggle `IWS_SKIP_LD_PATCH` (`66e460d`) to isolate `LD_LIBRARY_PATH` effects without changing other runtime settings.
- 2026-03-12 15:14 UTC - submitted isolated LD test `2789711` with `IWS_SKIP_LD_PATCH=1` and same manual-loop debug flags.
- 2026-03-12 15:14 UTC - `2789711` failed early during import (before training start) with `ImportError: ... GLIBC_2.38 not found` from OpenCV/cv2 (`libGLX.so.0`) when legacy LD patch is skipped.
- 2026-03-12 15:14 UTC - conclusion from LD-only test: current container startup in this repo still depends on the LD patch to avoid GLIBC/OpenCV import failure; this A/B did not reach backward path.
- 2026-03-12 15:23 UTC - committed/pushed manual-loop refinement (`990f8ca`) so Lightning hooks are only called in manual mode when `IWS_DEBUG_MANUAL_CALL_HOOKS=1`.
- 2026-03-12 15:24 UTC - submitted `2789735` with `IWS_DEBUG_MANUAL_TORCH_LOOP=1`, `IWS_DEBUG_MANUAL_CALL_HOOKS=0`, and legacy LD patch on (`IWS_SKIP_LD_PATCH=0`) to isolate pure manual backward path.
- 2026-03-12 15:25-15:27 UTC - `2789735` still hangs after full step-0 forward markers (`stage1_after_log`), with no `manual_torch debug step=` completion line even when hook calls are disabled.
- 2026-03-12 15:27 UTC - canceled `2789735` after capture; final Slurm state `CANCELLED`.

## Root cause

**`tracemalloc` deadlocks PyTorch's C++ autograd engine on ARM64/GH200.**

`tracemalloc.start()` (called in `on_train_start()`) hooks into every Python memory allocation via `PyMem_SetAllocator`. The C++ autograd engine uses a thread pool of worker threads that allocate Python objects (gradient tensors, etc.) and need the GIL. With tracemalloc active on ARM64, the trace hook + GIL acquisition pattern deadlocks the autograd worker threads, causing `loss.backward()` to hang forever with 0% GPU utilization — the engine gets stuck before launching any CUDA kernels.

This was present in the codebase from the initial commit (not a debugging artifact). It works on x86_64 but deadlocks on aarch64.

### How it was identified

1. `faulthandler.dump_traceback_later()` confirmed the hang was inside `_engine_run_backward` (C++ autograd engine), not Python code.
2. Component bisection (encoder-only backward) showed even a trivial 704-param `nn.Sequential` of Conv2d+SiLU hangs when the full model is loaded — ruling out computation graph complexity.
3. Setting `IWS_NO_TRACEMALLOC=1` to skip `tracemalloc.start()` **immediately resolved the hang** — encoder backward, decoder backward, and full `training_step` backward all completed.
4. Confirmation run (`2807429`): 3 full training steps completed in 16s with decreasing loss (`0.089 → 0.062 → 0.101`).

### Earlier red herrings

- cuDNN version mismatch (old container's 9.5.1 vs PyTorch's 9.10.2) — happened to coincide with the hang but was not the cause.
- LD_LIBRARY_PATH ordering — needed for OpenCV/GLIBC but unrelated to the backward hang.
- Attention backend selection (A100 vs GH200 path) — valid fix but not the backward hang cause.
- Lightning Trainer internals — ruled out by reproducing with a manual torch loop.

## Fixes applied

1. **tracemalloc disabled on ARM64** (`latent_world_model.py`): `on_train_start()` checks `platform.machine() == "aarch64"` and skips `tracemalloc.start()`. `training_step()` checks `is_tracing()` before taking snapshots.
2. **LD_LIBRARY_PATH ordering** (`slurm/iws_train_stage1_slurm.sh`): prepends PyTorch's bundled cuDNN/cuBLAS/torch libs, then container syslibs, then Apptainer-injected driver libs.
3. **Container base image** (`Dockerfile`): changed to `nvidia/cuda:12.6.3-devel-ubuntu22.04` (no system cuDNN — avoids version conflicts).
4. **OpenCV headless** (`requirements.txt`): `opencv-python-headless` / `opencv-contrib-python-headless` — no `libGLX.so` dependency, avoids GLIBC_2.38 mismatch.
5. **Attention backend** (`attention.py`): GH200 correctly uses math/efficient SDPA (not flash-attention-only A100 path).
6. **DataLoader** (`exp_base.py`): `prefetch_factor=1` only set when `num_workers > 0`.

## Job (full training run — OOM)
- job_id: `2808067`
- submitted: `2026-03-12 21:58 UTC`
- config: `isambard_train.yaml` (batch 16, 1M steps, ckpt every 2000, val every 6000)
- container: `interactive-world-sim_isambard-arm64.sif` (rebuilt with all fixes)
- note: first clean run after resolving tracemalloc deadlock. Training started and GPU hit 100% util, but OOM killed after ~1h.
- failure: `Detected 1 oom_kill event` — CPU memory exceeded 128G Slurm allocation (8 train + 16 val dataloader workers loading H5 data).
- fix: reduced workers to 4/4, bumped Slurm mem to 256G.

## Job (full training run — val_render crash)
- job_id: `2813013`
- submitted: `2026-03-13 07:10 UTC`
- config: `isambard_train.yaml` (batch 16, 1M steps, 4 workers, 256G mem)
- container: `interactive-world-sim_isambard-arm64.sif`
- note: training ran successfully for 6000 steps (2.5h, GPU 60-100% util, 0.80 it/s). Crashed at first validation (step 6000) in `on_validation_epoch_end` → `log_video` → `wandb.Video` requires `moviepy` and `imageio` which are not in the container.
- checkpoint saved: `epoch=0-step=6000.ckpt` (383MB)
- wandb synced run: `https://wandb.ai/pravsels/interactive_world_sim/runs/g3rkdc73`
- fix: set `val_render: False` in isambard config.

## Job (resumed from step 6000 — Hydra parse error)
- job_id: `2815001`
- submitted: `2026-03-13 09:42 UTC`
- failure: `mismatched input '=' expecting <EOF>` — Hydra parser choked on `=` signs in checkpoint filename (`epoch=0-step=6000.ckpt`).
- fix: quoted `LOAD_CKPT_PATH` value in slurm script.

## Job (resumed from step 6000)
- job_id: `2815251`
- submitted: `2026-03-13 10:25 UTC`
- config: `isambard_train.yaml` (batch 16, 1M steps, 4 workers, 256G mem, val_render off)
- resumed from: `epoch=0-step=6000.ckpt`
- outcome: failed in 15s with exit code `1:0` (early-launch failure), re-submitted immediately.

## Job (resumed from step 6000)
- job_id: `2815284`
- submitted: `2026-03-13 10:25 UTC`
- config: `isambard_train.yaml` (batch 16, 1M steps, 4 workers, 256G mem, val_render off)
- resumed from: `/workspace/outputs/2026-03-13/07-08-12/checkpoints/epoch=0-step=6000.ckpt`
- node: `nid010184`
- outcome: hit Slurm walltime limit (`TIMEOUT`) after `1-00:00:18`; apptainer step completed but batch step canceled on time limit.
- checkpoint saved: `/scratch/u6cr/pravsels.u6cr/interactive_world_sim/outputs/2026-03-13/10-25-30/checkpoints/epoch=0-step=64000.ckpt`
- wandb offline dir: `/scratch/u6cr/pravsels.u6cr/interactive_world_sim/outputs/2026-03-13/10-25-30/wandb/offline-run-20260313_102540-7gximny3`
- wandb synced run: `https://wandb.ai/pravsels/interactive_world_sim/runs/7gximny3`

## Results
- final step: `64000` (latest checkpoint before timeout)
- val_loss: `pending`
- training/rec_loss: `0.0037808 -> 0.0004945` over steps `6099 -> 64099` (min `0.0001879`)
- checkpoint: `/scratch/u6cr/pravsels.u6cr/interactive_world_sim/outputs/2026-03-13/10-25-30/checkpoints/epoch=0-step=64000.ckpt`
- wandb offline dir: `/scratch/u6cr/pravsels.u6cr/interactive_world_sim/outputs/2026-03-13/10-25-30/wandb/offline-run-20260313_102540-7gximny3`
- wandb synced run: `https://wandb.ai/pravsels/interactive_world_sim/runs/7gximny3`
- hf artifact repo: `https://huggingface.co/pravsels/interactive-world-sim-checkpoints`
- hf artifact folder: `stage1_front_cam_step64000/`

## Next
- resume from `epoch=0-step=64000.ckpt` in the next 1-day run.
- monitor next resumed job and capture first/last logged losses for this segment.
- related stage-2 run log: `runs/train_stage2_front_cam_2026-03-15.md`
