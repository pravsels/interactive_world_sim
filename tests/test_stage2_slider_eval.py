import numpy as np
import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inference" / "slider_control_utils.py"
_SPEC = importlib.util.spec_from_file_location("stage2_slider_eval", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec for {_SCRIPT_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_slider_delta_action = _MODULE.compute_slider_delta_action


def test_compute_slider_delta_action_steps_when_slider_changes():
    prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    curr = np.array([0.25, -0.5, 0.0], dtype=np.float32)

    action, should_step = compute_slider_delta_action(prev, curr, action_scale=2.0, epsilon=1e-6)

    assert should_step is True
    np.testing.assert_allclose(action, np.array([0.5, -1.0, 0.0], dtype=np.float32))


def test_compute_slider_delta_action_no_step_when_unchanged():
    prev = np.array([0.1, -0.2], dtype=np.float32)
    curr = np.array([0.1, -0.2], dtype=np.float32)

    action, should_step = compute_slider_delta_action(prev, curr, action_scale=1.0, epsilon=1e-6)

    assert should_step is False
    np.testing.assert_allclose(action, np.zeros_like(prev))


def test_compute_slider_delta_action_ignores_tiny_jitter():
    prev = np.array([0.0, 0.0], dtype=np.float32)
    curr = np.array([1e-7, -5e-7], dtype=np.float32)

    action, should_step = compute_slider_delta_action(prev, curr, action_scale=1.0, epsilon=1e-5)

    assert should_step is False
    np.testing.assert_allclose(action, np.zeros_like(prev))
