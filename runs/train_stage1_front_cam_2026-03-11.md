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
- job_id: `pending`
- submitted: `pending`
- node: `pending`
- submit command: `sbatch slurm/iws_train_stage1_slurm.sh`

## Status
- 2026-03-11 - prepared run log before first submission.

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
