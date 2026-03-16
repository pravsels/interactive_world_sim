from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_slider_delta_action(
    prev_slider: np.ndarray,
    curr_slider: np.ndarray,
    action_scale: float,
    epsilon: float,
) -> Tuple[np.ndarray, bool]:
    if prev_slider.shape != curr_slider.shape:
        raise ValueError(
            f"Shape mismatch for slider delta: {prev_slider.shape} vs {curr_slider.shape}"
        )
    delta = curr_slider - prev_slider
    if np.max(np.abs(delta)) <= float(epsilon):
        return np.zeros_like(delta), False
    return delta * float(action_scale), True
