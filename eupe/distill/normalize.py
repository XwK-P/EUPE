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
from torch import Tensor, nn


class FeatureNormalizer(nn.Module):
    """Frozen per-coordinate standardizer for one teacher + one token type.

    Args:
        dim: teacher feature dim for this token type.
    """

    def __init__(self, dim: int):
        super().__init__()
        # TODO: register_buffer("mean", zeros(dim)); register_buffer("std", ones(dim)).
        raise NotImplementedError("TODO: register frozen mean/std buffers")

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        """Copy estimated mean/std into the frozen buffers (called once after warmup)."""
        raise NotImplementedError("TODO: copy_ into buffers, clamp std away from 0")

    def forward(self, x: Tensor) -> Tensor:
        """Return (x - mean) / std, broadcasting over leading dims."""
        raise NotImplementedError("TODO: standardize x with frozen buffers")


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
    raise NotImplementedError("TODO: accumulate mean/var over 500 iters, build normalizers (paper 4.1)")
