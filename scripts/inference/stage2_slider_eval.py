#!/usr/bin/env python3
"""Quick interactive Stage-2 slider eval (imagined bot only, no real robot).

Quickstart:
python scripts/inference/stage2_slider_eval.py \
  --checkpoint checkpoints/stage2_front_cam_step20000/checkpoints/epoch=0-step=20000.ckpt \
  --config-snapshot checkpoints/stage2_front_cam_step20000/training_config_snapshot.yaml \
  --h5-path arx5_datasets_single_wan.h5 \
  --trajectory-idx 0 \
  --frame-idx 60

Behavior:
- Moving a slider advances the imagined rollout by one step.
- Action input is slider delta (current - previous slider value).
- If sliders are unchanged, action delta is zero and no auto-step happens.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
from scripts.inference.slider_control_utils import compute_slider_delta_action


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


def _decode_one(
    model: LatentWorldModel,
    latent_bchw: torch.Tensor,
    resolution: int,
    obs_key: str,
    image_normalizer: Any,
) -> np.ndarray:
    with torch.no_grad():
        out = render_img_cm(
            model,
            latent=latent_bchw,
            resolution=resolution,
            normalizer={obs_key: image_normalizer},
            num_views=1,
            batch_size=1,
        )
    img = out.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _read_slider_values(window: str, action_dim: int) -> np.ndarray:
    vals = np.zeros((action_dim,), dtype=np.float32)
    for i in range(action_dim):
        p = cv2.getTrackbarPos(f"a{i}", window)  # [0, 200]
        vals[i] = (float(p) - 100.0) / 100.0  # [-1, 1]
    return vals


def _step_latent(
    model: LatentWorldModel,
    latent_bchw: torch.Tensor,
    action_ba: torch.Tensor,
) -> torch.Tensor:
    # dynamics_forward expects action length >= history length + 1.
    # Build a tiny 2-step chunk: [zero_action, user_action_delta].
    zero = torch.zeros_like(action_ba)
    action_seq = torch.stack([zero, action_ba], dim=1)  # (1, 2, A)
    with torch.no_grad():
        pred = model.dynamics_forward(latent_bchw[:, None], action_seq)  # (1, 1, C, H, W)
    return pred[:, -1]


def _compose_frame(
    init_img: np.ndarray,
    pred_img: np.ndarray,
    selected_trajectory: str,
    frame_idx: int,
    step_idx: int,
    display_scale: int,
) -> np.ndarray:
    side = np.concatenate([init_img, pred_img], axis=1)
    canvas = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
    if display_scale > 1:
        canvas = cv2.resize(
            canvas,
            (canvas.shape[1] * display_scale, canvas.shape[0] * display_scale),
            interpolation=cv2.INTER_CUBIC,
        )
    cv2.putText(
        canvas,
        f"traj={selected_trajectory} frame={frame_idx} step={step_idx}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        canvas,
        "controls: move sliders=step (delta action), n=zero-step r=reset c=center q=quit",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    return canvas


def run(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}), replace=True)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

    cfg = OmegaConf.load(args.config_snapshot)
    cfg.algorithm.device = args.device
    cfg.algorithm.training_stage = 2
    cfg.algorithm.val_render = False
    cfg.dataset.h5_path = str(args.h5_path)
    # Local eval should not require the original stage-1 pretrain path from training.
    cfg.algorithm.load_ae = None

    obs_key = args.obs_key
    camera_dataset_key = _resolve_camera_dataset_key(cfg.dataset, obs_key)
    resolution = int(cfg.dataset.resolution)

    with h5py.File(args.h5_path, "r") as h5_file:
        if args.trajectory_idx is not None:
            selected_trajectory = f"trajectory_{args.trajectory_idx}"
        else:
            selected_trajectory = args.trajectory

        if selected_trajectory not in h5_file:
            raise KeyError(f"Trajectory '{selected_trajectory}' not found in {args.h5_path}")
        traj = h5_file[selected_trajectory]
        if camera_dataset_key not in traj:
            raise KeyError(
                f"Camera key '{camera_dataset_key}' not found in {selected_trajectory}"
            )
        n_frames = int(traj[camera_dataset_key].shape[0])
        if args.frame_idx < 0 or args.frame_idx >= n_frames:
            raise ValueError(
                f"--frame-idx {args.frame_idx} out of range for {selected_trajectory} "
                f"(valid: 0..{n_frames - 1})"
            )
        raw = traj[camera_dataset_key][int(args.frame_idx)]
        init_img = _center_crop_resize(_to_hwc_uint8(raw), resolution)

    device = torch.device(args.device)
    model = LatentWorldModel.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        cfg=cfg.algorithm,
        map_location=device,
    )
    model.eval()
    model.to(device)
    image_normalizer = get_image_range_normalizer().to(device)

    init_t = (
        torch.from_numpy(np.transpose(init_img.astype(np.float32) / 255.0, (2, 0, 1)))
        .unsqueeze(0)
        .to(device)
    )
    init_norm = image_normalizer.normalize(init_t)
    with torch.no_grad():
        init_latent = model.encoder_forward(init_norm)
    curr_latent = init_latent.clone()

    action_dim = int(cfg.algorithm.action_dim)
    window = "stage2_slider_eval"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI window is unavailable. This usually means headless OpenCV is installed "
            "(opencv-python-headless). Rebuild environment with GUI OpenCV (opencv-python) "
            "and run with X11 forwarding."
        ) from exc

    for i in range(action_dim):
        cv2.createTrackbar(f"a{i}", window, 100, 200, lambda _x: None)

    step_idx = 0
    prev_slider = _read_slider_values(window, action_dim)

    while True:
        curr_slider = _read_slider_values(window, action_dim)
        action_np, changed = compute_slider_delta_action(
            prev_slider=prev_slider,
            curr_slider=curr_slider,
            action_scale=args.action_scale,
            epsilon=args.slider_epsilon,
        )
        if changed:
            action_t = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
            curr_latent = _step_latent(model, curr_latent, action_t)
            prev_slider = curr_slider
            step_idx += 1

        pred_img = _decode_one(model, curr_latent, resolution, obs_key, image_normalizer)
        canvas = _compose_frame(
            init_img=init_img,
            pred_img=pred_img,
            selected_trajectory=selected_trajectory,
            frame_idx=args.frame_idx,
            step_idx=step_idx,
            display_scale=max(args.display_scale, 1),
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(max(1, int(1000 / max(args.fps, 1)))) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            zero_t = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
            curr_latent = _step_latent(model, curr_latent, zero_t)
            step_idx += 1
        elif key == ord("r"):
            curr_latent = init_latent.clone()
            step_idx = 0
            prev_slider = _read_slider_values(window, action_dim)
        elif key == ord("c"):
            for i in range(action_dim):
                cv2.setTrackbarPos(f"a{i}", window, 100)
            prev_slider = np.zeros((action_dim,), dtype=np.float32)

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Stage-2 slider eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-snapshot", type=Path, required=True)
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--trajectory", type=str, default="trajectory_0")
    parser.add_argument(
        "--trajectory-idx",
        type=int,
        default=None,
        help="If set, use trajectory_<idx> (overrides --trajectory).",
    )
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--obs-key", type=str, default="camera_1_color")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--slider-epsilon", type=float, default=1e-6)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--display-scale",
        type=int,
        default=4,
        help="Upscale display for readability.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
