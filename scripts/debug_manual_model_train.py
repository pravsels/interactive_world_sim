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
    if os.environ.get("IWS_DISABLE_CUDNN", "0") == "1":
        torch.backends.cudnn.enabled = False
        print("cuDNN DISABLED by IWS_DISABLE_CUDNN=1", flush=True)

    print("building experiment...", flush=True)
    exp = build_experiment(cfg, logger=None, ckpt_path=None)
    print("building algo...", flush=True)
    model = exp._build_algo()
    print("building training loader...", flush=True)
    train_loader = exp._build_training_loader()
    assert train_loader is not None, "Training dataloader is None."
    print("setting normalizer...", flush=True)
    if hasattr(model, "set_normalizer"):
        model.set_normalizer(train_loader.dataset.get_normalizer())  # type: ignore[attr-defined]
    print("moving model to device...", flush=True)
    model = model.to(device)
    model.train()
    model.log = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    model.log_dict = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    if os.environ.get("IWS_SKIP_ON_TRAIN_START", "0") != "1":
        print("on_train_start...", flush=True)
        model.on_train_start()
    else:
        import tracemalloc
        tracemalloc.start()
        model.tracemalloc_snapshot = tracemalloc.take_snapshot()
        print("SKIPPING on_train_start (tracemalloc started manually)", flush=True)
    print("configure_optimizers...", flush=True)
    optim_bundle = model.configure_optimizers()
    optimizer = optim_bundle["optimizer"]
    print("init done, starting component bisection", flush=True)
    import faulthandler, sys
    faulthandler.dump_traceback_later(90, repeat=False, file=sys.stdout, exit=True)

    # Test 1: encoder backward (704 params, trivial)
    dummy_img = torch.randn(1, 3, 224, 224, device=device)
    enc_out = model.encoder(dummy_img)
    enc_out.mean().backward()
    torch.cuda.synchronize()
    print(f"TEST1 encoder_backward OK shape={enc_out.shape}", flush=True)
    model.zero_grad()

    # Test 2: decoder backward (27.6M params)
    dummy_z = torch.randn(1, 4, 56, 56, device=device, requires_grad=True)
    dummy_noise = torch.randn(1, 3, 224, 224, device=device)
    dummy_t = torch.tensor([0.5], device=device)
    try:
        dec_out = model.decoder(dummy_noise, dummy_t, dummy_z)
        dec_out.mean().backward()
        torch.cuda.synchronize()
        print(f"TEST2 decoder_backward OK shape={dec_out.shape}", flush=True)
    except Exception as e:
        print(f"TEST2 decoder_backward FAILED: {e}", flush=True)
    model.zero_grad()

    # Test 3: dynamics backward (5.8M params)
    dummy_latent = torch.randn(1, 1, 4, 56, 56, device=device, requires_grad=True)
    dummy_action = torch.randn(1, 1, 7, device=device)
    try:
        dyn_out = model.dynamics(dummy_latent, dummy_action)
        dyn_out.mean().backward()
        torch.cuda.synchronize()
        print(f"TEST3 dynamics_backward OK shape={dyn_out.shape}", flush=True)
    except Exception as e:
        print(f"TEST3 dynamics_backward FAILED: {e}", flush=True)
    model.zero_grad()

    # Test 4: full training_step backward
    for step, batch in enumerate(train_loader):
        if step >= 1:
            break
        batch = _to_device(batch, device)
        model.zero_grad()
        out = model.training_step(batch, step)
        loss = out["loss"]
        print(f"TEST4 full forward_done loss={float(loss.detach().cpu()):.6f}", flush=True)
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        print("TEST4 full backward_done", flush=True)

    faulthandler.cancel_dump_traceback_later()
    print("ALL TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
