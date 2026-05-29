# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Multi-teacher feature-matching loss (paper §3.2, Eq. 4-7).

Per teacher i:
    L_i^c = cosine(z_cls, y_cls_norm)
    L_i^p = alpha * cosine(z_patch, y_patch_norm) + beta * smooth_l1(z_patch, y_patch_norm)
with alpha=0.9, beta=0.1. Total L = sum_i (L_i^c + L_i^p). Optional gamma multiplies the
DINOv3 teacher's patch term (Eq. 7). Spatial mismatch is fixed by bicubic 2D interpolation of
the smaller patch grid up to max(N_S, N_T). Lineage: AM-RADIO (cosine summary + cos/smooth-L1
spatial).
"""
from typing import Dict

import torch
from torch import Tensor, nn


class DistillationLoss(nn.Module):
    """Sum of per-teacher cls + patch feature-matching losses.

    Args:
        alpha: patch cosine weight (0.9).
        beta: patch smooth-L1 weight (0.1).
        dinov3_patch_gamma: extra multiplier on the DINOv3 teacher's patch loss (Eq. 7); 1.0 default.
        dinov3_teacher_name: which teacher key gamma applies to.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 0.1,
        dinov3_patch_gamma: float = 1.0,
        dinov3_teacher_name: str = "dinov3_hplus",
    ):
        super().__init__()
        # TODO: store hyperparameters.
        raise NotImplementedError("TODO: store alpha/beta/gamma/teacher name")

    @staticmethod
    def cosine_loss(z: Tensor, y: Tensor) -> Tensor:
        """Mean (1 - cosine_similarity) over the last dim (and tokens, for patches)."""
        raise NotImplementedError("TODO: 1 - F.cosine_similarity(z, y, dim=-1), then mean")

    @staticmethod
    def interpolate_patch_tokens(z: Tensor, y: Tensor) -> "tuple[Tensor, Tensor]":
        """Bicubic-interpolate the smaller of z/y patch grids up to max(N_S, N_T) so shapes match.

        z is [B, N_S, d], y is [B, N_T, d]; reshape to square grids, F.interpolate(mode='bicubic'),
        flatten back. See paper §3.1 (2D interpolation) — torchvision bicubic.
        """
        raise NotImplementedError("TODO: square-reshape, bicubic interpolate smaller -> larger")

    def patch_loss(self, z: Tensor, y: Tensor) -> Tensor:
        """alpha * cosine_loss + beta * smooth_l1, after spatial alignment."""
        raise NotImplementedError("TODO: align then alpha*cos + beta*smooth_l1 (Eq. 5)")

    def forward(
        self,
        adapted_student: Dict[str, Dict[str, Tensor]],
        teacher_normalized: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute the total loss and per-teacher breakdown.

        Args:
            adapted_student: {teacher_name: {"cls": z_cls, "patch": z_patch}} (adapter outputs).
            teacher_normalized: {teacher_name: {"cls": y_cls, "patch": y_patch}} (normalized teacher tokens).
        Returns:
            {"loss": scalar, "<name>_cls": ..., "<name>_patch": ...} for logging. (Eq. 6)
        """
        raise NotImplementedError("TODO: sum cls+patch losses over teachers, apply gamma (Eq. 6/7)")
