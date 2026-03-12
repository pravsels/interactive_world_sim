"""Standalone training smoke test — runs a few steps outside Lightning.

Usage (inside container on Isambard):
    python scripts/debug_manual_model_train.py
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

from interactive_world_sim.experiments import build_experiment

NUM_STEPS = int(os.environ.get("IWS_DEBUG_STEPS", "3"))


def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=False)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = str(repo_root / "configurations")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="isambard_train")

    with open_dict(cfg):
        cfg.experiment._name = "exp_latent_dyn"
        cfg.dataset._name = "arx5_h5_dataset"
        cfg.algorithm._name = "latent_world_model"
        cfg.debug = False
        cfg.experiment.training.data.num_workers = 0
        cfg.experiment.validation.data.num_workers = 0
        cfg.dataset.h5_path = os.environ.get("WAN_H5_PATH", "/mnt/wan_dataset.h5")

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    print(f"cudnn_version {torch.backends.cudnn.version()}", flush=True)

    exp = build_experiment(cfg, logger=None, ckpt_path=None)
    model = exp._build_algo()
    train_loader = exp._build_training_loader()
    assert train_loader is not None
    if hasattr(model, "set_normalizer"):
        model.set_normalizer(train_loader.dataset.get_normalizer())
    model = model.to(device)
    model.train()
    model.log = lambda *a, **kw: None
    model.log_dict = lambda *a, **kw: None
    model.on_train_start()
    optimizer = model.configure_optimizers()["optimizer"]

    start = time.time()
    for step, batch in enumerate(train_loader):
        if step >= NUM_STEPS:
            break
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, step)["loss"]
        print(f"step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
        loss.backward()
        optimizer.step()

    print(f"OK {NUM_STEPS} steps in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
