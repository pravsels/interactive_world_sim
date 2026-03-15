import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from imgaug import augmenters as iaa
from omegaconf import DictConfig
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.pytorch_util import dict_apply
from interactive_world_sim.utils.sampler import create_indices, get_val_mask

from .base_dataset import BaseImageDataset


def _trajectory_sort_key(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match is None:
        return (-1, name)
    return (int(match.group(1)), name)


def _running_stats_update(
    min_v: Optional[np.ndarray],
    max_v: Optional[np.ndarray],
    sum_v: Optional[np.ndarray],
    sumsq_v: Optional[np.ndarray],
    count: int,
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    arr = arr.astype(np.float64)
    if min_v is None:
        min_v = arr.min(axis=0)
        max_v = arr.max(axis=0)
        sum_v = arr.sum(axis=0)
        sumsq_v = np.square(arr).sum(axis=0)
    else:
        min_v = np.minimum(min_v, arr.min(axis=0))
        max_v = np.maximum(max_v, arr.max(axis=0))
        sum_v = sum_v + arr.sum(axis=0)
        sumsq_v = sumsq_v + np.square(arr).sum(axis=0)
    count += int(arr.shape[0])
    return min_v, max_v, sum_v, sumsq_v, count


def _running_stats_finalize(
    min_v: np.ndarray,
    max_v: np.ndarray,
    sum_v: np.ndarray,
    sumsq_v: np.ndarray,
    count: int,
) -> Dict[str, np.ndarray]:
    mean = sum_v / max(count, 1)
    var = np.maximum((sumsq_v / max(count, 1)) - np.square(mean), 0.0)
    std = np.sqrt(var)
    return {
        "min": min_v.astype(np.float32),
        "max": max_v.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


class Arx5H5Dataset(BaseImageDataset):
    """Stream trajectories directly from a single ARX5 HDF5 file.

    Camera convention for WAN-style ARX5 H5 files:
    - ``camera_0`` is wrist
    - ``camera_1`` is front
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.h5_path = cfg.h5_path
        self.shape_meta = cfg.shape_meta
        self.skip_frame = cfg.skip_frame
        self.sequence_length = cfg.horizon * cfg.skip_frame
        self.val_horizon = cfg.val_horizon * cfg.skip_frame
        self.pad_before = cfg.pad_before
        self.pad_after = cfg.pad_after
        self.skip_idx = cfg.skip_idx
        self.goal_sample = cfg.goal_sample
        self.seed = cfg.seed
        self.resolution = cfg.resolution
        self.action_key = cfg.action_key if "action_key" in cfg else "actions_delta"
        self.state_key = cfg.state_key if "state_key" in cfg else "states"
        self.camera_key_map = (
            dict(cfg.camera_key_map) if "camera_key_map" in cfg else dict()
        )
        self.lowdim_key_map = (
            dict(cfg.lowdim_key_map) if "lowdim_key_map" in cfg else dict()
        )
        self.stats_json_path = (
            str(cfg.stats_json_path)
            if "stats_json_path" in cfg and cfg.stats_json_path is not None
            else None
        )
        self._stats_json = self._load_stats_json(self.stats_json_path)
        self._stats_json_log_emitted = False

        self.aug_mode = cfg.aug_mode
        if self.aug_mode == "img_aug":
            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-30, 30),
                        mode="edge",
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    ),
                    iaa.MultiplyHueAndSaturation(
                        mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)
                    ),
                    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
                ]
            )
        elif self.aug_mode == "none":
            self.aug = None
        else:
            raise ValueError(f"Invalid augmentation mode: {self.aug_mode}")

        self.rgb_keys: List[str] = []
        self.depth_keys: List[str] = []
        self.lowdim_keys: List[str] = []
        selected_obs_keys = (
            list(cfg.obs_keys) if "obs_keys" in cfg and cfg.obs_keys is not None else None
        )
        if selected_obs_keys is not None:
            unknown_obs_keys = [
                key for key in selected_obs_keys if key not in self.shape_meta["obs"]
            ]
            if unknown_obs_keys:
                raise KeyError(
                    f"obs_keys contains keys missing from shape_meta.obs: {unknown_obs_keys}"
                )

        for key, attr in self.shape_meta["obs"].items():
            if selected_obs_keys is not None and key not in selected_obs_keys:
                continue
            key_type = attr.get("type", "low_dim")
            if key_type == "rgb":
                self.rgb_keys.append(key)
            elif key_type == "depth":
                self.depth_keys.append(key)
            elif key_type == "low_dim":
                self.lowdim_keys.append(key)

        self._h5: Optional[h5py.File] = None
        self._trajectory_names, self._episode_lengths = self._scan_trajectories()
        self.episode_ends = np.cumsum(self._episode_lengths, dtype=np.int64)
        self.n_episodes = len(self._trajectory_names)

        val_ratio = cfg.val_ratio if "val_ratio" in cfg else 0.0
        self.val_mask = get_val_mask(self.n_episodes, val_ratio=val_ratio, seed=self.seed)
        self.train_mask = ~self.val_mask
        if not np.any(self.train_mask):
            self.train_mask = np.ones_like(self.val_mask, dtype=bool)

        self.indices = create_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.sequence_length,
            episode_mask=self.train_mask,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _scan_trajectories(self) -> Tuple[List[str], np.ndarray]:
        with h5py.File(self.h5_path, "r") as h5_file:
            names = sorted(
                [k for k in h5_file.keys() if k.startswith("trajectory_")],
                key=_trajectory_sort_key,
            )
            if len(names) == 0:
                raise ValueError(f"No trajectory_* groups found in {self.h5_path}")
            lengths = []
            for name in names:
                if self.action_key not in h5_file[name]:
                    raise KeyError(
                        f"Trajectory {name} missing action key '{self.action_key}'."
                    )
                lengths.append(int(h5_file[name][self.action_key].shape[0]))
        return names, np.array(lengths, dtype=np.int64)

    def _resolve_obs_h5_key(self, obs_key: str) -> str:
        if obs_key in self.camera_key_map:
            return self.camera_key_map[obs_key]
        return obs_key

    def _resolve_lowdim_h5_key(self, lowdim_key: str) -> str:
        if lowdim_key in self.lowdim_key_map:
            return self.lowdim_key_map[lowdim_key]
        if lowdim_key == "joint_pos":
            return self.state_key
        return lowdim_key

    def _buffer_idx_to_epi_idx(self, buffer_idx: int) -> Tuple[int, int]:
        epi_idx = int(np.searchsorted(self.episode_ends, buffer_idx, side="right"))
        epi_start = self.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
        return epi_idx, int(buffer_idx - epi_start)

    def _read_traj_slice(
        self, traj_name: str, key: str, start: int, end: int
    ) -> np.ndarray:
        h5_file = self._get_h5()
        return h5_file[traj_name][key][start:end]

    def _read_traj_item(self, traj_name: str, key: str, idx: int) -> np.ndarray:
        h5_file = self._get_h5()
        return h5_file[traj_name][key][idx]

    def _pad_to_sequence(
        self, data: np.ndarray, sample_start: int, sample_end: int, sequence_length: int
    ) -> np.ndarray:
        if sample_start == 0 and sample_end == sequence_length:
            return data
        padded = np.zeros((sequence_length,) + data.shape[1:], dtype=data.dtype)
        if sample_start > 0:
            padded[:sample_start] = data[0]
        if sample_end < sequence_length:
            padded[sample_end:] = data[-1]
        padded[sample_start:sample_end] = data
        return padded

    def _build_sample(self, idx: int, sequence_length: int) -> Dict[str, np.ndarray]:
        indices = self.indices[idx * self.skip_idx]
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            int(indices[0]),
            int(indices[1]),
            int(indices[2]),
            int(indices[3]),
        )
        epi_idx, local_start = self._buffer_idx_to_epi_idx(buffer_start_idx)
        traj_name = self._trajectory_names[epi_idx]
        episode_end = int(self.episode_ends[epi_idx])

        is_early_stop = False
        rel_stop_idx = -1
        if self.goal_sample == "final":
            final_idx_global = episode_end - 1
        elif self.goal_sample == "intermediate":
            intermediate_start = min(buffer_end_idx, episode_end - 1)
            final_idx_global = int(np.random.randint(intermediate_start, episode_end))
        elif self.goal_sample == "aggressive":
            is_early_stop = np.random.rand() < 0.2
            if is_early_stop:
                rel_stop_idx = int(np.random.randint(0, sequence_length // self.skip_frame))
                final_idx_global = -1
            else:
                intermediate_start = min(buffer_end_idx, episode_end - 1)
                final_idx_global = int(np.random.randint(intermediate_start, episode_end))
        else:
            raise ValueError(f"Invalid goal sample: {self.goal_sample}")

        key_map: Dict[str, str] = {"action": self.action_key}
        for key in self.rgb_keys + self.depth_keys:
            key_map[key] = self._resolve_obs_h5_key(key)
        for key in self.lowdim_keys:
            key_map[key] = self._resolve_lowdim_h5_key(key)

        result: Dict[str, np.ndarray] = {}
        for model_key, h5_key in key_map.items():
            local_end = local_start + (buffer_end_idx - buffer_start_idx)
            seq_data = self._read_traj_slice(traj_name, h5_key, local_start, local_end)
            seq_data = self._pad_to_sequence(
                seq_data, sample_start_idx, sample_end_idx, sequence_length
            )
            if model_key == "action":
                inter_frames = sequence_length // self.skip_frame
                data_shape = list(seq_data.shape[1:])
                data_shape[0] = data_shape[0] * self.skip_frame
                seq_data = seq_data.reshape(inter_frames, self.skip_frame, *seq_data.shape[1:])
                seq_data = seq_data.reshape(-1, *data_shape)
            else:
                seq_data = seq_data[:: self.skip_frame]
            result[model_key] = seq_data

            if self.goal_sample == "aggressive" and is_early_stop:
                final_obs = seq_data[rel_stop_idx]
            else:
                _, final_local_idx = self._buffer_idx_to_epi_idx(final_idx_global)
                final_obs = self._read_traj_item(traj_name, h5_key, final_local_idx)
            result[f"{model_key}_final"] = final_obs

        result["is_early_stop"] = np.array(is_early_stop, dtype=np.bool_)
        result["rel_stop_idx"] = np.array(rel_stop_idx, dtype=np.int64)
        return result

    def _resize_rgb(self, imgs: np.ndarray, key: str) -> np.ndarray:
        shape = tuple(self.shape_meta["obs"][key]["shape"])
        c, h, w = shape
        out = []
        for img in imgs:
            cropped = center_crop(img, (h, w))
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
            out.append(resized)
        arr = np.stack(out, axis=0)
        assert arr[0].shape == (h, w, c)
        return arr

    def _resize_depth(self, imgs: np.ndarray, key: str) -> np.ndarray:
        shape = tuple(self.shape_meta["obs"][key]["shape"])
        c, h, w = shape
        out = []
        for img in imgs:
            cropped = center_crop(img, (h, w))
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
            out.append(resized)
        arr = np.stack(out, axis=0)[..., None]
        arr = np.clip(arr, 0, 1000).astype(np.uint16)
        assert arr[0].shape == (h, w, c)
        return arr

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        final_dict = dict()
        apply_aug = np.random.random() < 0.2 if self.aug_mode == "img_aug" else False

        for key in self.rgb_keys:
            obs_images = self._resize_rgb(sample[key], key).astype(np.uint8)
            final_images = self._resize_rgb(sample[f"{key}_final"][None], key)[0].astype(
                np.uint8
            )
            if apply_aug and self.aug is not None:
                aug_det = self.aug.to_deterministic()
                combined = [*obs_images, final_images]
                combined_aug = [aug_det.augment_image(img) for img in combined]
                obs_images = np.stack(combined_aug[:-1], axis=0)
                final_images = combined_aug[-1]
            obs_dict[key] = np.moveaxis(obs_images, -1, 1).astype(np.float32) / 255.0
            final_dict[key] = np.moveaxis(final_images, -1, 0).astype(np.float32) / 255.0

        for key in self.depth_keys:
            obs_depth = self._resize_depth(sample[key], key)
            final_depth = self._resize_depth(sample[f"{key}_final"][None], key)[0]
            obs_dict[key] = np.moveaxis(obs_depth, -1, 1).astype(np.float32) / 1000.0
            final_dict[key] = np.moveaxis(final_depth, -1, 0).astype(np.float32) / 1000.0

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            final_dict[key] = sample[f"{key}_final"].astype(np.float32)

        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "goal": dict_apply(final_dict, torch.from_numpy),
            "action": torch.from_numpy(sample["action"].astype(np.float32)),
            "is_early_stop": torch.from_numpy(np.array([sample["is_early_stop"]])),
            "rel_stop_idx": torch.from_numpy(np.array([sample["rel_stop_idx"]])),
        }
        return data

    def _build_stats_for_key(self, h5_key: str) -> Dict[str, np.ndarray]:
        h5_file = self._get_h5()
        min_v: Optional[np.ndarray] = None
        max_v: Optional[np.ndarray] = None
        sum_v: Optional[np.ndarray] = None
        sumsq_v: Optional[np.ndarray] = None
        count = 0
        for traj_name in self._trajectory_names:
            arr = h5_file[traj_name][h5_key][()]
            min_v, max_v, sum_v, sumsq_v, count = _running_stats_update(
                min_v, max_v, sum_v, sumsq_v, count, arr
            )
        assert min_v is not None and max_v is not None
        assert sum_v is not None and sumsq_v is not None
        return _running_stats_finalize(min_v, max_v, sum_v, sumsq_v, count)

    def _load_stats_json(self, path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
        if path is None:
            return None
        stats_path = Path(path)
        if not stats_path.exists():
            raise FileNotFoundError(f"stats_json_path not found: {stats_path}")
        with stats_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        parsed: Dict[str, np.ndarray] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                parsed[key] = np.asarray(value, dtype=np.float32)
        return parsed

    def _stats_from_json_prefix(self, prefix: str) -> Optional[Dict[str, np.ndarray]]:
        if self._stats_json is None:
            return None
        min_key = f"{prefix}_min"
        max_key = f"{prefix}_max"
        mean_key = f"{prefix}_mean"
        std_key = f"{prefix}_std"
        required = [min_key, max_key, mean_key, std_key]
        if not all(k in self._stats_json for k in required):
            return None
        return {
            "min": self._stats_json[min_key],
            "max": self._stats_json[max_key],
            "mean": self._stats_json[mean_key],
            "std": self._stats_json[std_key],
        }

    def _get_stats_for_h5_key(self, h5_key: str) -> Dict[str, np.ndarray]:
        if self._stats_json is not None:
            prefixes: List[str] = []
            if h5_key == self.action_key:
                # action_key is usually actions_delta in this dataset.
                prefixes.extend(["action", "actions_delta", "actions"])
            elif h5_key == self.state_key:
                prefixes.extend(["state", "states"])
            else:
                prefixes.append(h5_key)
            for prefix in prefixes:
                stat = self._stats_from_json_prefix(prefix)
                if stat is not None:
                    if not self._stats_json_log_emitted and self.stats_json_path is not None:
                        print(
                            f"[Arx5H5Dataset] using stats_json_path={self.stats_json_path} "
                            f"for normalizer stats (matched '{prefix}' for key '{h5_key}')",
                            flush=True,
                        )
                        self._stats_json_log_emitted = True
                    return stat
        return self._build_stats_for_key(h5_key)

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        action_stats = self._get_stats_for_h5_key(self.action_key)
        normalizer["action"] = get_range_normalizer_from_stat(action_stats)

        for key in self.lowdim_keys:
            stat = self._get_stats_for_h5_key(self._resolve_lowdim_h5_key(key))
            if key.endswith("quat") or key.endswith("vel") or key.endswith("pos"):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            else:
                normalizer[key] = get_identity_normalizer_from_stat(stat)

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.indices) // self.skip_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._build_sample(idx, sequence_length=self.sequence_length)
        return self._sample_to_data(sample)

    def get_validation_dataset(self) -> "BaseImageDataset":
        val_set = copy.copy(self)
        val_set.is_val = True
        val_set.indices = create_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.val_horizon,
            episode_mask=self.val_mask,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        return val_set
