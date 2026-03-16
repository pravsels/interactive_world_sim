#!/usr/bin/env python3
"""Quick Stage-2 replay eval: GT trajectory vs model rollout video.

This script:
1) Loads a Stage-2 checkpoint
2) Picks trajectory_* groups from a local H5
3) Rolls latent dynamics forward using recorded actions
4) Decodes imagined frames
5) Writes side-by-side GT vs imagined videos + simple metrics

Quickstart:
python scripts/inference/stage2_replay_eval.py \
  --checkpoint checkpoints/stage2_front_cam_step20000/checkpoints/epoch=0-step=20000.ckpt \
  --config-snapshot checkpoints/stage2_front_cam_step20000/training_config_snapshot.yaml \
  --h5-path arx5_datasets_single_wan.h5 \
  --output-dir outputs/stage2_replay_eval \
  --max-trajectories 2 --max-steps 200

Pick one trajectory explicitly:
python scripts/inference/stage2_replay_eval.py \
  --checkpoint checkpoints/stage2_front_cam_step20000/checkpoints/epoch=0-step=20000.ckpt \
  --config-snapshot checkpoints/stage2_front_cam_step20000/training_config_snapshot.yaml \
  --h5-path arx5_datasets_single_wan.h5 \
  --output-dir outputs/stage2_replay_eval_traj1 \
  --trajectory-idx 1 --start-frame 60 --max-steps 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.utils.normalizer import get_image_range_normalizer


def _trajectory_sort_key(name: str) -> Tuple[int, str]:
    suffix = name.split("_")[-1]
    if suffix.isdigit():
        return int(suffix), name
    return -1, name


def _resolve_camera_dataset_key(dataset_cfg: Any, obs_key: str) -> str:
    camera_key_map = getattr(dataset_cfg, "camera_key_map", None)
    if camera_key_map is None:
        return obs_key
    if obs_key in camera_key_map:
        return camera_key_map[obs_key]
    return obs_key


def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return np.clip(frame, 0, 255).astype(np.uint8)


def _center_crop_resize(frame_hwc: np.ndarray, resolution: int) -> np.ndarray:
    h, w = frame_hwc.shape[:2]
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    cropped = frame_hwc[top : top + crop, left : left + crop]
    return cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_AREA)


def _mse_mae_psnr(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))
    psnr = float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-12))))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def _decode_latents(
    model: LatentWorldModel,
    latents_btchw: torch.Tensor,
    resolution: int,
    obs_key: str,
    image_normalizer: Any,
    batch_size: int,
) -> np.ndarray:
    b, t, c, h, w = latents_btchw.shape
    flat = latents_btchw.reshape(b * t, c, h, w)
    bs = max(int(batch_size), 1)
    while True:
        try:
            with torch.no_grad():
                decoded = render_img_cm(
                    model,
                    latent=flat,
                    resolution=resolution,
                    normalizer={obs_key: image_normalizer},
                    num_views=1,
                    batch_size=bs,
                )
            if bs != batch_size:
                print(f"[decode] succeeded after OOM fallback with batch_size={bs}", flush=True)
            break
        except torch.OutOfMemoryError:
            if flat.device.type == "cuda":
                torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(bs // 2, 1)
            print(f"[decode] OOM, retrying with batch_size={bs}", flush=True)
    return decoded.detach().cpu().permute(0, 2, 3, 1).numpy().reshape(b, t, resolution, resolution, 3)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}), replace=True)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

    cfg = OmegaConf.load(args.config_snapshot)
    cfg.dataset.h5_path = str(args.h5_path)
    cfg.algorithm.device = args.device
    cfg.algorithm.training_stage = 2
    cfg.algorithm.val_render = False
    # Local eval should not require the original stage-1 pretrain path from training.
    cfg.algorithm.load_ae = None

    obs_key = args.obs_key
    camera_dataset_key = _resolve_camera_dataset_key(cfg.dataset, obs_key)
    action_dataset_key = args.action_key or str(cfg.dataset.action_key)
    resolution = int(cfg.dataset.resolution)

    device = torch.device(args.device)
    model = LatentWorldModel.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        cfg=cfg.algorithm,
        map_location=device,
    )
    model.eval()
    model.to(device)

    image_normalizer = get_image_range_normalizer().to(device)

    out_dir = args.output_dir
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "config_snapshot": str(args.config_snapshot),
        "h5_path": str(args.h5_path),
        "obs_key": obs_key,
        "camera_dataset_key": camera_dataset_key,
        "action_dataset_key": action_dataset_key,
        "trajectories": [],
    }

    with h5py.File(args.h5_path, "r") as h5_file:
        traj_names = sorted(
            [k for k in h5_file.keys() if k.startswith("trajectory_")],
            key=_trajectory_sort_key,
        )
        if not traj_names:
            raise ValueError(f"No trajectory_* groups found in {args.h5_path}")

        if args.trajectory_idx is not None:
            selected_name = f"trajectory_{args.trajectory_idx}"
            if selected_name not in h5_file:
                raise KeyError(
                    f"Requested --trajectory-idx {args.trajectory_idx} "
                    f"but '{selected_name}' not found in {args.h5_path}"
                )
            selected = [selected_name]
        else:
            rng = np.random.default_rng(args.seed)
            n_pick = min(args.max_trajectories, len(traj_names))
            picked = rng.choice(len(traj_names), size=n_pick, replace=False)
            selected = [traj_names[int(i)] for i in picked]

        print(f"selected_trajectories: {selected}", flush=True)
        for traj_name in tqdm(selected, desc="trajectories", unit="traj"):
            stage_pbar = tqdm(total=4, desc=f"{traj_name} stages", unit="stage", leave=False)
            group = h5_file[traj_name]
            if camera_dataset_key not in group:
                raise KeyError(f"Missing camera key '{camera_dataset_key}' in {traj_name}")
            if action_dataset_key not in group:
                raise KeyError(f"Missing action key '{action_dataset_key}' in {traj_name}")

            raw_frames = group[camera_dataset_key]
            raw_actions = group[action_dataset_key]
            total_n = min(int(raw_frames.shape[0]), int(raw_actions.shape[0]))
            if args.start_frame < 0 or args.start_frame >= total_n:
                raise ValueError(
                    f"--start-frame {args.start_frame} out of range for {traj_name} "
                    f"(valid: 0..{total_n - 1})"
                )
            n = min(total_n - args.start_frame, args.max_steps)
            if n < 2:
                continue

            gt_uint8 = np.stack(
                [
                    _center_crop_resize(
                        _to_hwc_uint8(raw_frames[args.start_frame + i]), resolution
                    )
                    for i in range(n)
                ],
                axis=0,
            )
            gt_np = gt_uint8.astype(np.float32) / 255.0
            action_np = np.asarray(
                raw_actions[args.start_frame : args.start_frame + n], dtype=np.float32
            )
            stage_pbar.set_postfix_str("prep")
            stage_pbar.update(1)

            # Encode first frame as rollout seed.
            first = torch.from_numpy(np.transpose(gt_np[0], (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
            first_norm = image_normalizer.normalize(first)
            with torch.no_grad():
                z0 = model.encoder_forward(first_norm)  # (1, C, H, W)

            act = torch.from_numpy(action_np).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, T, A)
            act_norm = model.normalizer["action"].normalize(act)

            with torch.no_grad():
                z_roll = model.dynamics_forward(z0[:, None], act_norm)  # (1, T-1, C, H, W)
            z_full = torch.cat([z0[:, None], z_roll], dim=1)[:, :n]  # (1, T, C, H, W)
            stage_pbar.set_postfix_str("rollout")
            stage_pbar.update(1)

            pred_np = _decode_latents(
                model=model,
                latents_btchw=z_full,
                resolution=resolution,
                obs_key=obs_key,
                image_normalizer=image_normalizer,
                batch_size=args.decode_batch_size,
            )[0]
            stage_pbar.set_postfix_str("decode")
            stage_pbar.update(1)

            m = _mse_mae_psnr(gt_np, pred_np)

            # side-by-side video
            video_path = video_dir / f"{traj_name}_gt_vs_pred.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps),
                (resolution * 2 * args.display_scale, resolution * args.display_scale),
            )
            for i in tqdm(range(n), desc=f"{traj_name} frames", unit="frame", leave=False):
                gt_img = (np.clip(gt_np[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                pr_img = (np.clip(pred_np[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                side = np.concatenate([gt_img, pr_img], axis=1)
                side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
                if args.display_scale > 1:
                    side_bgr = cv2.resize(
                        side_bgr,
                        (resolution * 2 * args.display_scale, resolution * args.display_scale),
                        interpolation=cv2.INTER_CUBIC,
                    )
                scaled_res = resolution * args.display_scale
                # Top labels
                cv2.putText(
                    side_bgr,
                    "REAL",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    side_bgr,
                    "IMAGINED",
                    (scaled_res + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                # Small run tag at bottom-left
                info_text = f"{traj_name} | step {args.start_frame + i}"
                text_y = max(24, scaled_res - 12)
                cv2.putText(
                    side_bgr,
                    info_text,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(side_bgr)
            writer.release()
            stage_pbar.set_postfix_str("write")
            stage_pbar.update(1)
            stage_pbar.close()

            metrics["trajectories"].append(
                {
                    "trajectory": traj_name,
                    "start_frame": int(args.start_frame),
                    "num_steps": n,
                    "video": str(video_path),
                    "metrics": m,
                }
            )
            print(f"[saved] {traj_name} -> {video_path}", flush=True)

    if not metrics["trajectories"]:
        raise RuntimeError("No valid trajectories evaluated.")

    agg = {
        "mse_mean": float(np.mean([t["metrics"]["mse"] for t in metrics["trajectories"]])),
        "mae_mean": float(np.mean([t["metrics"]["mae"] for t in metrics["trajectories"]])),
        "psnr_mean": float(np.mean([t["metrics"]["psnr"] for t in metrics["trajectories"]])),
    }
    metrics["aggregate"] = agg

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"output_dir": str(out_dir), "aggregate": agg}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Stage-2 replay eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-snapshot", type=Path, required=True)
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obs-key", type=str, default="camera_1_color")
    parser.add_argument(
        "--action-key",
        type=str,
        default=None,
        help="Override action dataset key (default: from config dataset.action_key, usually actions_delta).",
    )
    parser.add_argument("--max-trajectories", type=int, default=3)
    parser.add_argument(
        "--trajectory-idx",
        type=int,
        default=None,
        help="If set, evaluate only trajectory_<idx> (overrides random max-trajectories sampling).",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start rollout from this frame index within each selected trajectory.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--display-scale",
        type=int,
        default=4,
        help="Upscale factor for the side-by-side video before drawing text (improves text clarity).",
    )
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
