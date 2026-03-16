# train_stage3 - front cam WAN h5 decoder finetuning

## Mode
- run_type: training
- objective: train stage-3 autoencoder finetuning on WAN-format ARX5 data, initialized from the Stage-2 baseline checkpoint.

## Config
- script: `slurm/iws_train_stage3_slurm.sh`
- config: `configurations/isambard_train_stage3.yaml`
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_wan224.h5`
- dataset stats: `/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_stats.json`
- key settings:
  - dataset: `arx5_h5_dataset`
  - obs_keys: `['camera_1_color']` (front only)
  - action key: `actions_delta` (dim=7)
  - training stage: `3` (autoencoder finetuning)
  - train batch size: `16`
  - train/val dataloader workers: `4/4`
  - horizon: `1` (`val_horizon: 200`)
  - init checkpoint (stage-2 baseline): `/workspace/outputs/2026-03-15/17-44-19/checkpoints/epoch=0-step=20000.ckpt`
  - sampling strategy: `terminal_only`
  - noise loss weighting: `uniform`
  - wandb mode: `offline`
  - checkpoint every: `10000` train steps
  - walltime: `1-00:00:00`

## Job
- job_id: `2896349`
- submitted: `2026-03-16 13:51 UTC`
- node: `nid010111`
- submit command: `sbatch slurm/iws_train_stage3_slurm.sh`

## Status
- 2026-03-16 13:51 UTC - run submitted and started on `nid010111`.
- startup confirmed:
  - cuDNN check passed: `cudnn_version=91002`
  - dataset stats bind path in use: `[Arx5H5Dataset] using stats_json_path=/mnt/wan_dataset_stats.json ...`
  - output dir: `/workspace/outputs/2026-03-16/13-52-11`
  - W&B offline run id: `z2vt6qah`
- scheduler state checks:
  - `squeue`: `RUNNING` on `nid010111`
  - `sacct`: `RUNNING`, exit code `0:0`
- telemetry sample:
  - `nvidia-smi`: `GPU util 100%`, `VRAM 67861/97871 MiB`, `power 414.51 W` (sampled at `13:55:08` UTC)
- 2026-03-16 21:14 UTC - run failed during validation dataloader with:
  - `ValueError: could not broadcast input array from shape (199,7) into shape (0,7)`
  - stack trace points to `interactive_world_sim/datasets/latent_dynamics/arx5_h5_dataset.py::_pad_to_sequence` from validation loop.
- root cause identified:
  - `get_validation_dataset()` builds indices with `val_horizon` but copied dataset still sampled with train `sequence_length` (`horizon`).
  - with `horizon=1` and `val_horizon=200`, val indices and sampling length diverged, causing invalid slice placement.
- fix prepared and validated:
  - patched `get_validation_dataset()` to set `val_set.sequence_length = val_set.val_horizon`.
  - reproduced failure on synthetic h5 before patch; validation sample succeeds after patch (`action` shape `(200, 7)`).
- 2026-03-16 22:03 UTC - resubmitted after fix:
  - `git pull --ff-only origin main` on Isambard updated repo to commit `ba80a84`.
  - new job submitted: `sbatch slurm/iws_train_stage3_slurm.sh` -> job id `2906576`.

## Results
- state: `failed`
- slurm job: `2896349`
- runtime window: `2026-03-16T13:51:57Z` -> `2026-03-16T21:14:04Z` (~`7h22m`)
- failure stage: `validation dataloader` at first validation pass (`step 30000`)
- error class: `ValueError` (sequence padding mismatch)
- checkpoint path: `not reached for this segment`

## Next
- monitor new `slurm-2906576.out/.err` through first validation boundary.
- sync W&B offline run once a healthy segment completes.
