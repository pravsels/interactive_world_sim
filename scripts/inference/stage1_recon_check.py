#!/usr/bin/env python3
"""Quick reconstruction quality check on a local ARX5 H5 file.

This script:
1) Loads a checkpoint
2) Samples frames from trajectory_* groups in an H5 file
3) Encodes + decodes those frames
4) Saves side-by-side GT/reconstruction images
5) Writes simple metrics (MSE, MAE, PSNR) to JSON

Quickstart (5 visual examples total):
python scripts/inference/stage1_recon_check.py \
  --checkpoint checkpoints/stage1_front_cam_step64000/checkpoints/epoch=0-step=64000.ckpt \
  --config-snapshot checkpoints/stage1_front_cam_step64000/training_config_snapshot.yaml \
  --h5-path arx5_datasets_single_wan.h5 \
  --output-dir outputs/local_stage1_recon_check_5 \
    --frames-per-trajectory 5 --max-trajectories 5
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
    """Convert CHW/HWC frame to HWC uint8."""
    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (H,W,C) or (C,H,W), got {frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _center_crop_resize(frame_hwc: np.ndarray, resolution: int) -> np.ndarray:
    h, w = frame_hwc.shape[:2]
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    cropped = frame_hwc[top : top + crop, left : left + crop]
    resized = cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return resized


def _collect_frame_samples(
    h5_path: Path,
    camera_dataset_key: str,
    max_trajectories: int,
    frames_per_trajectory: int,
    resolution: int,
    seed: int,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5_file:
        traj_names = sorted(
            [k for k in h5_file.keys() if k.startswith("trajectory_")],
            key=_trajectory_sort_key,
        )
        if not traj_names:
            raise ValueError(f"No trajectory_* groups found in {h5_path}")
        rng = np.random.default_rng(seed)
        n_pick = min(max_trajectories, len(traj_names))
        picked_idx = rng.choice(len(traj_names), size=n_pick, replace=False)
        traj_names = [traj_names[int(i)] for i in picked_idx]

        for traj_name in traj_names:
            if camera_dataset_key not in h5_file[traj_name]:
                raise KeyError(
                    f"Trajectory {traj_name} missing camera key '{camera_dataset_key}'."
                )
            arr = h5_file[traj_name][camera_dataset_key]
            n = int(arr.shape[0])
            if n == 0:
                continue
            frame_idxs = np.linspace(0, n - 1, num=min(frames_per_trajectory, n), dtype=int)
            for frame_idx in frame_idxs.tolist():
                frame_raw = arr[frame_idx]
                frame_hwc = _to_hwc_uint8(frame_raw)
                frame_hwc = _center_crop_resize(frame_hwc, resolution)
                samples.append(
                    {
                        "trajectory": traj_name,
                        "frame_idx": int(frame_idx),
                        "rgb_hwc_uint8": frame_hwc,
                    }
                )
    if not samples:
        raise ValueError("No frames were collected from the H5 file.")
    return samples


def _batch_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """GT/pred in [0,1], shape (N,H,W,C)."""
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))
    psnr = float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-12))))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def _save_visuals(
    out_dir: Path,
    samples: List[Dict[str, Any]],
    gt: np.ndarray,
    pred: np.ndarray,
) -> None:
    gt_dir = out_dir / "gt"
    pred_dir = out_dir / "recon"
    side_dir = out_dir / "side_by_side"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        stem = f"{i:04d}_{sample['trajectory']}_f{sample['frame_idx']:05d}"
        gt_img = (np.clip(gt[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        pred_img = (np.clip(pred[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        side = np.concatenate([gt_img, pred_img], axis=1)
        cv2.imwrite(str(gt_dir / f"{stem}.png"), cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(pred_dir / f"{stem}.png"), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(str(side_dir / f"{stem}.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Match main.py resolver behavior for config interpolation.
    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}), replace=True)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

    cfg = OmegaConf.load(args.config_snapshot)
    cfg.dataset.h5_path = str(args.h5_path)
    cfg.algorithm.training_stage = args.training_stage
    cfg.algorithm.device = args.device
    cfg.algorithm.val_render = True
    # Local checkpoint eval should not depend on historical pretrain paths.
    cfg.algorithm.load_ae = None

    obs_key = args.obs_key
    camera_dataset_key = _resolve_camera_dataset_key(cfg.dataset, obs_key)
    resolution = int(cfg.dataset.resolution)

    samples = _collect_frame_samples(
        h5_path=args.h5_path,
        camera_dataset_key=camera_dataset_key,
        max_trajectories=args.max_trajectories,
        frames_per_trajectory=args.frames_per_trajectory,
        resolution=resolution,
        seed=args.seed,
    )

    # N,H,W,C uint8 -> N,C,H,W float in [0,1]
    gt_np = np.stack([s["rgb_hwc_uint8"] for s in samples], axis=0).astype(np.float32) / 255.0
    gt_t = torch.from_numpy(np.transpose(gt_np, (0, 3, 1, 2)))

    device = torch.device(args.device)
    model = LatentWorldModel.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        cfg=cfg.algorithm,
        map_location=device,
    )
    model.eval()
    model.to(device)

    image_normalizer = get_image_range_normalizer().to(device)
    gt_t = gt_t.to(device=device, dtype=torch.float32)
    gt_norm = image_normalizer.normalize(gt_t)

    with torch.no_grad():
        z = model.encoder_forward(gt_norm)
        pred_t = render_img_cm(
            model,
            latent=z,
            resolution=resolution,
            normalizer={obs_key: image_normalizer},
            num_views=1,
            batch_size=args.decode_batch_size,
        )

    pred_np = pred_t.detach().cpu().permute(0, 2, 3, 1).numpy()
    metrics = _batch_metrics(gt_np, pred_np)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_visuals(out_dir, samples, gt_np, pred_np)

    detailed = []
    per_frame_err = np.mean((gt_np - pred_np) ** 2, axis=(1, 2, 3))
    for i, sample in enumerate(samples):
        detailed.append(
            {
                "index": i,
                "trajectory": sample["trajectory"],
                "frame_idx": sample["frame_idx"],
                "mse": float(per_frame_err[i]),
            }
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "config_snapshot": str(args.config_snapshot),
        "h5_path": str(args.h5_path),
        "obs_key": obs_key,
        "camera_dataset_key": camera_dataset_key,
        "num_samples": len(samples),
        "aggregate_metrics": metrics,
        "per_frame": detailed,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    print(json.dumps({"output_dir": str(out_dir), "aggregate_metrics": metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint reconstruction check")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Stage-1 checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config-snapshot",
        type=Path,
        required=True,
        help="Exact Hydra config snapshot used for the checkpoint.",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        required=True,
        help="Local ARX5 H5 file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write visuals and metrics.",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="camera_1_color",
        help="Observation key to evaluate (default: camera_1_color).",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=5,
        help="How many random trajectories to sample (default: 5 for quick visual check).",
    )
    parser.add_argument(
        "--frames-per-trajectory",
        type=int,
        default=1,
        help="How many frames to sample per trajectory (default: 1).",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=32,
        help="Batch size used in decoder sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda or cpu).",
    )
    parser.add_argument(
        "--training-stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Model training stage for checkpoint loading (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for trajectory/frame sampling reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
