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

    # Set names that are normally injected in main.py via Hydra runtime choices.
    with open_dict(cfg):
        cfg.experiment._name = "exp_latent_dyn"
        cfg.dataset._name = "arx5_h5_dataset"
        cfg.algorithm._name = "latent_world_model"
        cfg.debug = False
        cfg.experiment.training.data.num_workers = 0
        cfg.experiment.validation.data.num_workers = 0
        cfg.dataset.h5_path = os.environ.get("WAN_H5_PATH", "/mnt/wan_dataset.h5")

    assert torch.cuda.is_available(), "CUDA is required for this debug script."
    device = torch.device("cuda")
    print("device", torch.cuda.get_device_name(0), flush=True)
    print("CUDA_LAUNCH_BLOCKING", os.environ.get("CUDA_LAUNCH_BLOCKING", "unset"), flush=True)  # noqa: E501
    print("cudnn_version", torch.backends.cudnn.version(), flush=True)

    exp = build_experiment(cfg, logger=None, ckpt_path=None)
    model = exp._build_algo()

    train_loader = exp._build_training_loader()
    assert train_loader is not None, "Training dataloader is None."
    if hasattr(model, "set_normalizer"):
        model.set_normalizer(train_loader.dataset.get_normalizer())  # type: ignore[attr-defined]

    model = model.to(device)
    model.train()

    # Avoid Lightning logger/trainer requirements in manual loop mode.
    model.log = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    model.log_dict = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    model.on_train_start()

    optim_bundle = model.configure_optimizers()
    optimizer = optim_bundle["optimizer"]

    # Phase 1: test autograd BEFORE any model forward
    probe_pre = torch.tensor(1.0, device=device, requires_grad=True)
    (probe_pre * 1.0).backward()
    print("phase1_pre_forward_probe OK", flush=True)

    # Phase 2: test Conv2d backward with model's encoder
    dummy_img = torch.randn(1, 3, 224, 224, device=device)
    enc_out = model.encoder(dummy_img)
    enc_out.mean().backward()
    print(f"phase2_encoder_backward OK shape={enc_out.shape}", flush=True)
    model.zero_grad()

    # Phase 3: full training_step forward + backward
    start = time.time()
    for step, batch in enumerate(train_loader):
        if step >= 1:
            break
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model.training_step(batch, step)
        assert isinstance(out, dict) and "loss" in out, "training_step did not return loss"
        loss = out["loss"]
        print(f"step={step} forward_done loss={float(loss.detach().cpu()):.6f}", flush=True)
        torch.cuda.synchronize()
        print(f"step={step} pre_backward_sync_ok", flush=True)
        loss.backward()
        torch.cuda.synchronize()
        print(f"step={step} backward_done", flush=True)
        optimizer.step()
        print(f"step={step} optimizer_step_done", flush=True)

    print(f"MANUAL_TORCH_MODEL_LOOP_OK total_s={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
