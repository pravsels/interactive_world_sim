# train_stage2 - front cam WAN h5 dynamics

## Mode
- run_type: training
- objective: train stage-2 latent dynamics on WAN-format ARX5 data using the stage-1 checkpoint initialization.

## Config
- script: `slurm/iws_train_stage2_slurm.sh`
- config: `configurations/isambard_train_stage2.yaml`
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_wan224.h5`
- dataset stats: `/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_stats.json`
- key settings:
  - dataset: `arx5_h5_dataset`
  - obs_keys: `['camera_1_color']` (front only)
  - action key: `actions_delta` (dim=7)
  - training stage: `2` (latent dynamics)
  - training precision: `16-mixed`
  - training batch size: `32`
  - train/val dataloader workers: `8/8`
  - horizon: `10` (`val_horizon: 200`)
  - stage-1 init checkpoint: `/workspace/outputs/2026-03-13/10-25-30/checkpoints/epoch=0-step=64000.ckpt`
  - sampling strategy: `terminal_only`
  - noise loss weighting: `uniform`
  - wandb mode: `offline`
  - checkpoint every: `10000` train steps
  - walltime: `1-00:00:00`

## Job
- job_id: `2874725`
- submitted: `2026-03-15 17:21 UTC`
- node: `nid010020`
- submit command: `sbatch slurm/iws_train_stage2_slurm.sh`

## Job (throughput-tuned restart)
- job_id: `2874967`
- submitted: `2026-03-15 17:36 UTC`
- config change: switched to `experiment.training.batch_size=32` and `experiment.training.precision=16-mixed`
- node: `nid010631`
- outcome: canceled quickly to clear and relaunch.

## Job (throughput-tuned run)
- job_id: `2875003`
- submitted: `2026-03-15 17:39 UTC`
- config: `isambard_train_stage2.yaml` (`batch_size=32`, `precision=16-mixed`)
- node: `nid010653`
- startup confirms AMP: `Using 16bit Automatic Mixed Precision (AMP)`
- startup confirms stats JSON: `[Arx5H5Dataset] using stats_json_path=/mnt/wan_dataset_stats.json ...`
- output dir: `/workspace/outputs/2026-03-15/17-39-17`

## Job (throughput-tuned run, workers increase)
- job_id: `2875089`
- submitted: `2026-03-15 17:44 UTC`
- config: `isambard_train_stage2.yaml` (`batch_size=32`, `precision=16-mixed`, `num_workers=8`)
- node: `nid010008`
- startup confirms AMP: `Using 16bit Automatic Mixed Precision (AMP)`
- startup confirms stats JSON: `[Arx5H5Dataset] using stats_json_path=/mnt/wan_dataset_stats.json ...`
- output dir: `/workspace/outputs/2026-03-15/17-44-19`

## Status
- 2026-03-15 17:21 UTC - run submitted and started on `nid010020`.
- startup confirmed:
  - stats-json bind path exported: `WAN_STATS_JSON_CONTAINER_PATH=/mnt/wan_dataset_stats.json`
  - dataset log confirms stats usage: `[Arx5H5Dataset] using stats_json_path=/mnt/wan_dataset_stats.json ...`
  - model initialized and trainer started (GPU detected, stage-2 module summary printed).
- current scheduler status: `RUNNING` (`squeue` / `sacct`).
- output log files:
  - `slurm-2874725.out`
  - `slurm-2874725.err`
- run output dir (container path): `/workspace/outputs/2026-03-15/17-21-51`
- 2026-03-15 17:36 UTC - run `2874725` canceled to move to a higher-throughput config.
- 2026-03-15 17:36 UTC - submitted `2874967` with tuned config (`batch_size=32`, `16-mixed`) and canceled shortly after to relaunch cleanly.
- 2026-03-15 17:39 UTC - submitted `2875003` with tuned config (`batch_size=32`, `16-mixed`) on `nid010653`.
- 2026-03-15 17:44 UTC - run `2875003` canceled to test higher dataloader throughput.
- 2026-03-15 17:44 UTC - submitted `2875089` with `num_workers=8/8` on `nid010008`.
- telemetry sample for `2875089` after startup: VRAM climbs to `39162 MiB` with bursty GPU util (`100%, 0%, 47%, 29%` over successive 5s samples).
- 2026-03-16 01:39 UTC - job `2875089` finished with scheduler state `COMPLETED`, elapsed `07:55:12`, exit code `0:0`.
- final slurm progress line before exit: `Epoch 0/-2 ... 23015/40172 ... 0.77 it/s`.
- training emitted `NaN in gradient of module.out.1.weight`; run appears to stop early for numerical instability (not walltime or max_steps).
- synced W&B run: `https://wandb.ai/pravsels/interactive_world_sim/runs/7skk0qh6`

## Results
- state: `completed` (job `2875089`, ended early due to NaN-gradient instability)
- training loss (`training/loss`, W&B): `0.014964337 -> 3.7573023e-05` (min `2.3631907e-05`)
- logged global steps (W&B): `99 -> 22999` (230 points)
- validation metrics: `pending`
- checkpoint path: `/scratch/u6cr/pravsels.u6cr/interactive_world_sim/outputs/2026-03-15/17-44-19/checkpoints/epoch=0-step=20000.ckpt`
- wandb offline run id: `7skk0qh6`
- wandb synced run: `https://wandb.ai/pravsels/interactive_world_sim/runs/7skk0qh6`
- hf artifact repo: `https://huggingface.co/pravsels/interactive-world-sim-checkpoints`
- hf artifact folder: `stage2_front_cam_step20000/`

## Verdict
- practical stage-2 learning appears to have plateaued for this run segment before instability.
- freeze this checkpoint (`step=20000`) as the current stage-2 baseline rather than immediately continuing training.

## Next
- use `epoch=0-step=20000.ckpt` for downstream evaluation/integration.
- only resume stage-2 training if downstream metrics show a clear need for further improvement.
