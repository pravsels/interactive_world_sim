# eval - stage3 rollout quality (step 60000)

## Provenance
- checkpoint: `checkpoints/stage3_front_cam_step60000/checkpoints/epoch=0-step=60000.ckpt`
- source run log: `runs/train_stage3_front_cam_2026-03-16.md`
- config_snapshot: `checkpoints/stage3_front_cam_step60000/training_config_snapshot.yaml`
- dataset: `arx5_datasets_single_wan.h5` at `repo root`

## Job
- job_id: n/a (local container eval)
- submitted/start: `2026-03-17`
- start_human: `Mar 17, 2026`
- end: `2026-03-17`
- end_human: `Mar 17, 2026`
- runtime: `~47s` (stage2_replay_eval single trajectory)
- node: local docker container

## Metrics
- script: `scripts/inference/stage2_replay_eval.py`
- output_dir: `outputs/stage3_replay_eval_step60000`
- aggregate:
  - `mse_mean`: `0.2616567015647888`
  - `mae_mean`: `0.4297126829624176`
  - `psnr_mean`: `5.822681376094439`
- video: `outputs/stage3_replay_eval_step60000/videos/trajectory_0_gt_vs_pred.mp4`

## Qualitative
- stage1 recon check with the same stage3 checkpoint looked good (decoder reconstruction quality acceptable).
- replay rollout quality is severely degraded (strong noisy / speckled artifacts).
- slider eval with the same stage3 checkpoint shows the same failure mode after rollout steps.
- this behavior is much worse than the stage2 baseline checkpoint used earlier.

## Verdict
- verdict: stage3 checkpoint at step 60000 has good stage1-style reconstruction, but both stage2 replay and stage2 slider checks produce garbled outputs; not usable for rollout/control-time imagination quality.

## Next
- keep stage2 checkpoint for replay/slider usage.
- use stage3 checkpoints only for recon-focused checks unless rollout-alignment training is added.
- if stage3 rollout quality is a goal, add a dynamics-latent decode alignment objective before next stage3 run.
