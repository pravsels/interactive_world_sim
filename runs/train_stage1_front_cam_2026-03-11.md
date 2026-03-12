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

## Results
- final step: `pending`
- val_loss: `pending`
- checkpoint: `pending`
- wandb offline dir: `pending` (sync later with `wandb sync`)

## Next
- submit run and record job id + node.
- monitor `squeue`, `slurm-<jobid>.out`, and step/loss checkpoints here.
- if interrupted by walltime, resume with:
  - `sbatch --export=ALL,LOAD_CKPT_PATH=/scratch/.../outputs/.../checkpoints/<ckpt>.ckpt slurm/iws_train_stage1_slurm.sh`

## Debug commands used
- GPU telemetry sample (memory pinned vs compute idle):
  - `srun --jobid=2733840 --overlap /bin/bash -lc 'nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits'`
