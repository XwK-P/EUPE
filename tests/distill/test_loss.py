import torch
from eupe.distill.loss import DistillationLoss

def test_cosine_loss_bounds():
    v = torch.randn(4, 10)
    assert torch.allclose(DistillationLoss.cosine_loss(v, v), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(DistillationLoss.cosine_loss(v, -v), torch.tensor(2.0), atol=1e-6)

def test_interpolate_aligns_to_larger_grid():
    z = torch.randn(2, 4, 5)    # 2x2 grid
    y = torch.randn(2, 16, 5)   # 4x4 grid
    z2, y2 = DistillationLoss.interpolate_patch_tokens(z, y)
    assert z2.shape == (2, 16, 5) and y2.shape == (2, 16, 5)

def test_forward_assembles_loss_dict_and_applies_gamma():
    loss = DistillationLoss(alpha=0.9, beta=0.1, dinov3_patch_gamma=2.0, dinov3_teacher_name="d")
    adapted = {"d": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)}}
    tnorm   = {"d": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)}}
    out = loss(adapted, tnorm)
    assert "loss" in out and "d_cls" in out and "d_patch" in out
    assert out["loss"].requires_grad is False or out["loss"].dim() == 0
    # gamma path: doubling gamma doubles the patch contribution
    base = DistillationLoss(0.9, 0.1, 1.0, "d")(adapted, tnorm)
    torch.testing.assert_close(out["d_patch"], 2.0 * base["d_patch"])
