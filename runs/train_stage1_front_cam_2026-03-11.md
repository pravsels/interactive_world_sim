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
