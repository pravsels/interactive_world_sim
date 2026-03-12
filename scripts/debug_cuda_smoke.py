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

    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    y = (x @ x).mean()
    y.backward()
    print("matmul_backward_ok", bool(torch.isfinite(x.grad).all()))

    q = torch.randn(2, 4, 128, 64, device="cuda", requires_grad=True)
    k = torch.randn(2, 4, 128, 64, device="cuda", requires_grad=True)
    v = torch.randn(2, 4, 128, 64, device="cuda", requires_grad=True)
    out = F.scaled_dot_product_attention(q, k, v)
    loss = out.float().mean()
    loss.backward()
    print("sdpa_backward_ok", bool(torch.isfinite(q.grad).all()))

    # TF32 matmul backward (matches QKVAttention einsum pattern)
    torch.set_float32_matmul_precision("high")
    a = torch.randn(32, 64, 196, device="cuda", requires_grad=True)
    b = torch.randn(32, 64, 196, device="cuda", requires_grad=True)
    w = torch.einsum("bct,bcs->bts", a, b)
    w = torch.softmax(w.float(), dim=-1).to(a.dtype)
    print("einsum_attn_forward_ok", flush=True)
    w.sum().backward()
    print("einsum_attn_backward_ok", bool(torch.isfinite(a.grad).all()))

    # Conv2d backward test (cuDNN path)
    print("cudnn_enabled", torch.backends.cudnn.enabled)

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
    out = conv(img)
    loss = out.float().mean()
    print("conv_forward_ok", flush=True)
    loss.backward()
    print("conv_backward_ok", bool(all(torch.isfinite(p.grad).all() for p in conv.parameters())))

    # Same test with cuDNN disabled
    torch.backends.cudnn.enabled = False
    conv2 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.SiLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1, stride=2),
    ).cuda()
    img2 = torch.randn(16, 3, 224, 224, device="cuda")
    out2 = conv2(img2)
    loss2 = out2.float().mean()
    print("conv_no_cudnn_forward_ok", flush=True)
    loss2.backward()
    print("conv_no_cudnn_backward_ok", bool(all(torch.isfinite(p.grad).all() for p in conv2.parameters())))

    print("SMOKE_TEST_DONE")


if __name__ == "__main__":
    main()
