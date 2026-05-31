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


def test_estimate_statistics_forwards_teacher_at_native_resolution():
    # M19: the warmup must resize to each teacher's native_resolution (matching training), not feed
    # the raw crop resolution — otherwise frozen stats for PE teachers (native 448) are measured at
    # the student crop size (e.g. 256) and mismatch training.
    seen = {}

    class ResTeacher(torch.nn.Module):
        embed_dim = 4
        native_resolution = 96

        def forward(self, img):
            seen["hw"] = (int(img.shape[-2]), int(img.shape[-1]))
            b = img.shape[0]
            return {"cls": torch.zeros(b, 4), "patch": torch.zeros(b, 3, 4)}

    loader = ([torch.randn(2, 3, 64, 64)] for _ in range(1))  # crop res 64 != native 96
    estimate_teacher_statistics({"t": ResTeacher()}, loader, n_iters=1)
    assert seen["hw"] == (96, 96)
