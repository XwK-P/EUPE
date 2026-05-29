# tests/distill/test_normalize.py
import torch
from eupe.distill.normalize import FeatureNormalizer, estimate_teacher_statistics

def test_normalizer_buffers_frozen_and_standardize():
    n = FeatureNormalizer(8)
    assert not any(p.requires_grad for p in n.parameters())   # only buffers, no params
    torch.testing.assert_close(n(torch.zeros(3, 8)), torch.zeros(3, 8))   # mean0/std1 identity
    n.set_stats(torch.full((8,), 2.0), torch.full((8,), 4.0))
    torch.testing.assert_close(n(torch.full((3, 8), 6.0)), torch.ones(3, 8))  # (6-2)/4

def test_set_stats_clamps_zero_std():
    n = FeatureNormalizer(4); n.set_stats(torch.zeros(4), torch.zeros(4))
    assert torch.all(n.std > 0)

def test_estimate_statistics_recovers_distribution():
    torch.manual_seed(0)
    mu, sd = 5.0, 3.0
    class DummyTeacher(torch.nn.Module):
        embed_dim = 8
        def forward(self, img):
            b = img.shape[0]
            return {"cls": mu + sd*torch.randn(b, 8), "patch": mu + sd*torch.randn(b, 6, 8)}
    loader = ([torch.randn(64, 3, 16, 16)] for _ in range(200))   # img content irrelevant to dummy
    norms = estimate_teacher_statistics({"t": DummyTeacher()}, loader, n_iters=200)
    assert torch.allclose(norms["t"]["cls"].mean, torch.full((8,), mu), atol=0.3)
    assert torch.allclose(norms["t"]["cls"].std,  torch.full((8,), sd), atol=0.3)
