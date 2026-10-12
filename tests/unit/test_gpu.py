import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_gpu_bf16_matmul() -> None:
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    assert y.shape == (1024, 1024)
    assert torch.isfinite(y).all()
