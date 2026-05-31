# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Teacher feature normalization (paper §3.3).

Standardize each teacher's outputs per coordinate: (x - mean) / std, separately for cls and
patch tokens, per teacher. Stats are estimated ONCE over ~500 iterations before training, then
FROZEN. Simpler than RADIO PHI-S (radio/feature_normalizer.py) — no rotation matrix — which
avoids a per-step cross-GPU all-gather and lets batch size scale across nodes.
"""
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eupe import distributed

# Floor applied to per-coordinate std so that division never blows up on (near-)constant
# coordinates (report §6.3).
_STD_EPS = 1e-6


class FeatureNormalizer(nn.Module):
    """Frozen per-coordinate standardizer for one teacher + one token type.

    Args:
        dim: teacher feature dim for this token type.
    """

    # Ported from refs/RADIO/radio/feature_normalizer.py:FeatureNormalizer — divergence:
    # dropped the learned rotation matrix `tx` (PHI-S); store a diagonal mean/std only so
    # standardization is a pure per-coordinate affine and needs no cross-GPU all-gather at
    # train time (report §6.3).
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        """Copy estimated mean/std into the frozen buffers (called once after warmup)."""
        self.mean.copy_(mean)
        self.std.copy_(std)
        # Guard against zero/near-zero std on constant coordinates (report §6.3).
        self.std.clamp_(min=_STD_EPS)

    def forward(self, x: Tensor) -> Tensor:
        """Return (x - mean) / std, broadcasting over leading dims."""
        # mean/std are 1-D [dim]; they broadcast over any number of leading dims (e.g. [B, dim]
        # for cls tokens or [B, N, dim] for patch tokens).
        return (x - self.mean) / self.std


def estimate_teacher_statistics(
    teachers: Dict[str, nn.Module],
    data_loader: Iterable,
    n_iters: int = 500,
) -> Dict[str, Dict[str, "FeatureNormalizer"]]:
    """Run each frozen teacher over n_iters batches; accumulate per-coordinate mean/std for cls
    and patch tokens; return {teacher_name: {"cls": FeatureNormalizer, "patch": FeatureNormalizer}}.

    Paper §4.1: "crude centering ... measuring per-coordinate mean and variance during 500
    iterations before training." Run under torch.no_grad(); accumulate in fp32.
    """
    # Ported from refs/RADIO/radio/feature_normalizer.py — divergence: RADIO solves for a PHI-S
    # rotation; here we only accumulate streaming fp32 sum/sumsq to recover a diagonal
    # mean/std (report §6.3). Each statistic is summed over both the batch and (for patch
    # tokens) the token dimension, leaving a per-coordinate [dim] estimate.

    # Per teacher × token-type running accumulators (all fp32): sum, sum-of-squares, count.
    sums: Dict[str, Dict[str, Tensor]] = {}
    sumsqs: Dict[str, Dict[str, Tensor]] = {}
    counts: Dict[str, Dict[str, Tensor]] = {}

    def _accumulate(name: str, token_type: str, feats: Tensor) -> None:
        # Flatten everything but the feature dim, then reduce over the leading axis so that a
        # patch tensor [B, N, dim] contributes B*N samples and a cls tensor [B, dim] contributes B.
        feats = feats.detach().to(torch.float32).reshape(-1, feats.shape[-1])
        s = feats.sum(dim=0)
        ss = (feats * feats).sum(dim=0)
        n = torch.tensor(float(feats.shape[0]), dtype=torch.float32, device=feats.device)
        if name not in sums:
            sums[name], sumsqs[name], counts[name] = {}, {}, {}
        if token_type not in sums[name]:
            sums[name][token_type] = torch.zeros_like(s)
            sumsqs[name][token_type] = torch.zeros_like(ss)
            counts[name][token_type] = torch.zeros((), dtype=torch.float32, device=feats.device)
        sums[name][token_type] += s
        sumsqs[name][token_type] += ss
        counts[name][token_type] += n

    data_iter = iter(data_loader)
    with torch.no_grad():
        for _ in range(n_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            images = _extract_images(batch)
            for name, teacher in teachers.items():
                device = next((p.device for p in teacher.parameters()), images.device)
                # Resize to the teacher's native resolution BEFORE the forward, matching the
                # training-time DistillationMetaArch.get_teacher_outputs, so the frozen normalizer
                # stats are measured at the SAME resolution the teacher actually sees during
                # distillation (paper §3.3). Without this, PE teachers (native 448) would be measured
                # at the student/crop resolution (e.g. 256) and the frozen stats would not match training.
                teacher_images = _resize_to_native(images.to(device), teacher)
                out = teacher(teacher_images)
                for token_type in ("cls", "patch"):
                    _accumulate(name, token_type, out[token_type])

    normalizers: Dict[str, Dict[str, FeatureNormalizer]] = {}
    for name in sums:
        normalizers[name] = {}
        for token_type in ("cls", "patch"):
            total_sum = sums[name][token_type]
            total_sumsq = sumsqs[name][token_type]
            count = counts[name][token_type]
            # All-reduce across ranks so every process derives identical (global) stats.
            if distributed.is_enabled():
                torch.distributed.all_reduce(total_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(total_sumsq, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

            mean = total_sum / count
            var = total_sumsq / count - mean * mean
            std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_(min=_STD_EPS)

            norm = FeatureNormalizer(mean.shape[0]).to(mean.device)
            norm.set_stats(mean, std)
            normalizers[name][token_type] = norm
    return normalizers


def _resize_to_native(images: Tensor, teacher) -> Tensor:
    """Bicubic-resize images to the teacher's native_resolution (no-op if absent or already there).

    Mirrors DistillationMetaArch.get_teacher_outputs (bicubic, align_corners=False) so the normalizer
    warmup measures each teacher at the same resolution training feeds it.
    """
    res = getattr(teacher, "native_resolution", None)
    if res is None or (images.shape[-1] == res and images.shape[-2] == res):
        return images
    return F.interpolate(images, size=(int(res), int(res)), mode="bicubic", align_corners=False)


def _extract_images(batch) -> Tensor:
    """Pull the image tensor out of whatever the data loader yields.

    Accepts a bare tensor, a (images, ...) tuple/list, or a dict carrying the images under a
    common key. This keeps the estimator agnostic to the distillation loader's batch schema.
    """
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("collated_global_crops", "images", "image", "img"):
            if key in batch:
                return batch[key]
        # Fall back to the first tensor value in the dict.
        for value in batch.values():
            if isinstance(value, Tensor):
                return value
        raise KeyError("estimate_teacher_statistics: no image tensor found in batch dict")
    if isinstance(batch, (tuple, list)):
        return batch[0]
    raise TypeError(f"estimate_teacher_statistics: unsupported batch type {type(batch)!r}")
