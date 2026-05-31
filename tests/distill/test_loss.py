import torch
import torch.nn.functional as F
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
    # total is a finite 0-dim scalar equal to the single teacher's cls + (gamma-scaled) patch term.
    # (Previously this line asserted `requires_grad is False or dim()==0`, a tautology — dim()==0 always
    # holds for the scalar total, so it tested nothing.)
    assert out["loss"].dim() == 0 and torch.isfinite(out["loss"])
    torch.testing.assert_close(out["loss"], out["d_cls"] + out["d_patch"])
    # gamma path: doubling gamma doubles the patch contribution
    base = DistillationLoss(0.9, 0.1, 1.0, "d")(adapted, tnorm)
    torch.testing.assert_close(out["d_patch"], 2.0 * base["d_patch"])


def test_patch_loss_uses_0p9_cosine_plus_0p1_smoothl1():
    # Eq.5: L_p = 0.9*cos + 0.1*smooth_l1. Guards against swapping/altering the weights (these
    # defaults were previously asserted by NO test, so a swap passed the whole suite).
    loss = DistillationLoss()
    assert loss.alpha == 0.9 and loss.beta == 0.1
    torch.manual_seed(0)
    z = torch.randn(2, 9, 7)
    y = torch.randn(2, 9, 7)  # same token count -> interpolation is a no-op, so weights are isolated
    expected = 0.9 * DistillationLoss.cosine_loss(z, y) + 0.1 * F.smooth_l1_loss(z, y)
    torch.testing.assert_close(loss.patch_loss(z, y), expected)


def test_total_loss_is_sum_over_teachers():
    # Eq.6: L = sum_i (L_i^c + L_i^p), summed over all teachers.
    loss = DistillationLoss()
    torch.manual_seed(0)
    adapted = {
        "a": {"cls": torch.randn(2, 5), "patch": torch.randn(2, 9, 5)},
        "b": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)},
    }
    tnorm = {
        "a": {"cls": torch.randn(2, 5), "patch": torch.randn(2, 9, 5)},
        "b": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)},
    }
    out = loss(adapted, tnorm)  # neither teacher is the dinov3 teacher, so gamma never applies
    expected = out["a_cls"] + out["a_patch"] + out["b_cls"] + out["b_patch"]
    torch.testing.assert_close(out["loss"], expected)


def test_gamma_scales_only_the_dinov3_patch_term():
    # Eq.7: gamma multiplies ONLY the DINOv3 teacher's patch loss — not its cls, not other teachers.
    g = 3.0
    lg = DistillationLoss(dinov3_patch_gamma=g, dinov3_teacher_name="dv3")
    l1 = DistillationLoss(dinov3_patch_gamma=1.0, dinov3_teacher_name="dv3")
    torch.manual_seed(0)
    adapted = {
        "dv3": {"cls": torch.randn(2, 5), "patch": torch.randn(2, 9, 5)},
        "other": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)},
    }
    tnorm = {
        "dv3": {"cls": torch.randn(2, 5), "patch": torch.randn(2, 9, 5)},
        "other": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)},
    }
    a = lg(adapted, tnorm)
    b = l1(adapted, tnorm)
    torch.testing.assert_close(a["dv3_patch"], g * b["dv3_patch"])  # scaled
    torch.testing.assert_close(a["dv3_cls"], b["dv3_cls"])          # cls untouched
    torch.testing.assert_close(a["other_patch"], b["other_patch"])  # other teacher untouched
    torch.testing.assert_close(a["other_cls"], b["other_cls"])
