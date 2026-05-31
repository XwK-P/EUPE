# tests/distill/test_normalize.py
import torch
import torch.nn as nn

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


def test_estimate_statistics_cycles_pyramid_scales():
    # Stage-3 fix: when pyramid_scales is given, the warmup must measure the teacher across those scales
    # (cycled per step), NOT only its native_resolution — otherwise the frozen stats are measured at 256
    # while the proxy is actually run at {256,384,512} during multi-resolution training.
    seen = []

    class ResTeacher(torch.nn.Module):
        embed_dim = 4
        native_resolution = 256

        def forward(self, img):
            seen.append(int(img.shape[-1]))
            b = img.shape[0]
            return {"cls": torch.zeros(b, 4), "patch": torch.zeros(b, 3, 4)}

    loader = ([torch.randn(2, 3, 512, 512)] for _ in range(6))  # loader feeds the max pyramid crop
    estimate_teacher_statistics({"t": ResTeacher()}, loader, n_iters=6, pyramid_scales=[256, 384, 512])
    # 6 steps over a 3-scale pyramid cycle through 256,384,512 twice — never only native 256.
    assert seen == [256, 384, 512, 256, 384, 512]


def test_normalizer_estimates_per_coordinate_stats():
    # M22: normalization must be PER-COORDINATE. The existing recovery test uses i.i.d. coords, so a
    # global/scalar-stat bug would pass it. Use DISTINCT per-coordinate mean/std and assert both that
    # they are recovered and that the per-coordinate spread is preserved (not collapsed to one value).
    torch.manual_seed(0)
    mean = torch.tensor([0.0, 10.0, -5.0, 100.0])
    std = torch.tensor([1.0, 2.0, 0.5, 20.0])

    class PerCoordTeacher(torch.nn.Module):
        embed_dim = 4

        def forward(self, img):
            b = img.shape[0]
            return {
                "cls": mean + std * torch.randn(b, 4),
                "patch": mean + std * torch.randn(b, 5, 4),
            }

    loader = ([torch.randn(256, 3, 8, 8)] for _ in range(80))
    norms = estimate_teacher_statistics({"t": PerCoordTeacher()}, loader, n_iters=80)
    torch.testing.assert_close(norms["t"]["cls"].mean, mean, atol=0.5, rtol=0)
    torch.testing.assert_close(norms["t"]["cls"].std, std, atol=0.5, rtol=0)
    # A global-scalar implementation would give equal stds across coords (ratio ~1); per-coordinate
    # stats keep the 0.5..20 spread (ratio ~40).
    assert norms["t"]["cls"].std.max() > 5 * norms["t"]["cls"].std.min()


def test_only_teacher_outputs_are_normalized():
    # M21 / paper §3.3 invariant: compute_losses normalizes the TEACHER tokens and passes the STUDENT
    # tokens through the adapters WITHOUT normalization. We build a bare meta-arch (bypassing __init__,
    # which would load external teachers + a meta-device student) and exercise compute_losses directly.
    from eupe.distill.adapters import AdapterHeadSet
    from eupe.train.distill_meta_arch import DistillationMetaArch

    student_dim, teacher_dim, hidden = 6, 5, 8
    meta = DistillationMetaArch.__new__(DistillationMetaArch)
    nn.Module.__init__(meta)
    meta.adapters = AdapterHeadSet(student_dim, [("t", teacher_dim)], hidden)
    meta.normalizers = nn.ModuleDict(
        {"t": nn.ModuleDict({"cls": FeatureNormalizer(teacher_dim), "patch": FeatureNormalizer(teacher_dim)})}
    )
    meta.normalizers["t"]["cls"].set_stats(torch.full((teacher_dim,), 3.0), torch.ones(teacher_dim))
    meta.normalizers["t"]["patch"].set_stats(torch.full((teacher_dim,), 3.0), torch.ones(teacher_dim))

    captured = {}

    # Recorder stands in for the loss so we can inspect exactly what compute_losses passes it.
    # (Assigned directly — never registered as a submodule — so it stays a plain attribute.)
    def recorder(adapted_student, teacher_normalized):
        captured.update(student=adapted_student, teacher=teacher_normalized)
        return {"loss": torch.zeros(())}

    meta.loss = recorder

    student_cls = torch.randn(2, student_dim)
    student_patch = torch.randn(2, 4, student_dim)
    teacher_out = {"t": {"cls": torch.full((2, teacher_dim), 3.0), "patch": torch.full((2, 4, teacher_dim), 3.0)}}
    meta.compute_losses(student_cls, student_patch, teacher_out)

    # Teacher tokens ARE normalized: (3 - 3) / 1 == 0.
    torch.testing.assert_close(captured["teacher"]["t"]["cls"], torch.zeros(2, teacher_dim))
    torch.testing.assert_close(captured["teacher"]["t"]["patch"], torch.zeros(2, 4, teacher_dim))
    # Student tokens are NOT normalized — exactly the adapter output (no mean subtraction).
    expected = meta.adapters(student_cls, student_patch)
    torch.testing.assert_close(captured["student"]["t"]["cls"], expected["t"]["cls"])
    torch.testing.assert_close(captured["student"]["t"]["patch"], expected["t"]["patch"])
