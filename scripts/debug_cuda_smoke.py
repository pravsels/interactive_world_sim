import torch
import torch.nn.functional as F


def main() -> None:
    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device_count", torch.cuda.device_count())

    if not torch.cuda.is_available():
        print("SMOKE_TEST_DONE_NO_CUDA")
        return

    print("device0", torch.cuda.get_device_name(0))
    print("cudnn_available", torch.backends.cudnn.is_available())
    print("cudnn_version", torch.backends.cudnn.version())

    torch.set_float32_matmul_precision("high")

    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    y = (x @ x).mean()
    y.backward()
    print("matmul_backward_ok", bool(torch.isfinite(x.grad).all()))

    q = torch.randn(2, 4, 128, 64, device="cuda", requires_grad=True)
    k = torch.randn(2, 4, 128, 64, device="cuda", requires_grad=True)
    v = torch.randn(2, 4, 128, 64, device="cuda", requires_grad=True)
    out = F.scaled_dot_product_attention(q, k, v)
    out.float().mean().backward()
    print("sdpa_backward_ok", bool(torch.isfinite(q.grad).all()))

    conv = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.SiLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1, stride=2),
        torch.nn.SiLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.SiLU(),
        torch.nn.Conv2d(128, 128, 3, padding=1, stride=2),
    ).cuda()
    img = torch.randn(16, 3, 224, 224, device="cuda")
    conv(img).float().mean().backward()
    print("conv_backward_ok", bool(all(torch.isfinite(p.grad).all() for p in conv.parameters())))

    print("SMOKE_TEST_DONE")


if __name__ == "__main__":
    main()
