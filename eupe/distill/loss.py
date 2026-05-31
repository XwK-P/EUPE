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
import math
from typing import Dict

import torch
import torch.nn.functional as F
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
        self.alpha = alpha
        self.beta = beta
        self.dinov3_patch_gamma = dinov3_patch_gamma
        self.dinov3_teacher_name = dinov3_teacher_name

    @staticmethod
    def cosine_loss(z: Tensor, y: Tensor) -> Tensor:
        """Mean (1 - cosine_similarity) over the last dim (and tokens, for patches)."""
        # Ported from refs/RADIO/examples/mode_switching.py:215 (1 - F.cosine_similarity(...).mean())
        # — divergence: reduce over the embedding dim (-1) then mean over all remaining dims.
        return (1.0 - F.cosine_similarity(z, y, dim=-1)).mean()

    @staticmethod
    def interpolate_patch_tokens(z: Tensor, y: Tensor) -> "tuple[Tensor, Tensor]":
        """Bicubic-interpolate the smaller of z/y patch grids up to max(N_S, N_T) so shapes match.

        z is [B, N_S, d], y is [B, N_T, d]; reshape to square grids, F.interpolate(mode='bicubic'),
        flatten back. See paper §3.1 (2D interpolation) — torchvision bicubic.
        """
        n_s = z.shape[1]
        n_t = y.shape[1]
        n_target = max(n_s, n_t)
        z = DistillationLoss._resize_grid(z, n_target)
        y = DistillationLoss._resize_grid(y, n_target)
        return z, y

    @staticmethod
    def _resize_grid(t: Tensor, n_target: int) -> Tensor:
        """Bicubic-resize [B, N, d] tokens to n_target tokens via a square grid; no-op if already there."""
        b, n, d = t.shape
        if n == n_target:
            return t
        side = int(round(math.sqrt(n)))
        target_side = int(round(math.sqrt(n_target)))
        # Fail loudly (not with a confusing deep reshape error) if a teacher/student ever emits a
        # non-square patch grid. All EUPE encoders emit square grids at the paper resolutions (register
        # tokens are stripped from x_norm_patchtokens; ConvNeXt grids are square), so this never fires today.
        assert side * side == n, f"_resize_grid expects a square patch grid, got N={n}"
        assert target_side * target_side == n_target, f"_resize_grid expects a square target grid, got N={n_target}"
        # [B, N, d] -> [B, d, sqrt(N), sqrt(N)]
        grid = t.transpose(1, 2).reshape(b, d, side, side)
        grid = F.interpolate(grid, size=(target_side, target_side), mode="bicubic", align_corners=False)
        # [B, d, sqrt(Nt), sqrt(Nt)] -> [B, Nt, d]
        return grid.reshape(b, d, target_side * target_side).transpose(1, 2)

    def patch_loss(self, z: Tensor, y: Tensor) -> Tensor:
        """alpha * cosine_loss + beta * smooth_l1, after spatial alignment.

        smooth_l1 uses PyTorch's default Huber transition (beta=1.0). The paper (Eq. 5) says only
        "smooth L1" and does not specify the transition point, so 1.0 is a defensible default; note the
        cosine term is reduced per-token and the smooth-L1 term element-wise, matching the AM-RADIO
        lineage but meaning the alpha/beta weighting is relative to those two (paper-unspecified) reductions.
        """
        z, y = self.interpolate_patch_tokens(z, y)
        return self.alpha * self.cosine_loss(z, y) + self.beta * F.smooth_l1_loss(z, y)

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
        # Report §6.2: total = sum_i (L_i^c + L_i^p); the dinov3 teacher's patch term is scaled by gamma (Eq. 7).
        out: Dict[str, Tensor] = {}
        total = None
        for name, student in adapted_student.items():
            teacher = teacher_normalized[name]
            cls_loss = self.cosine_loss(student["cls"], teacher["cls"])
            patch_loss = self.patch_loss(student["patch"], teacher["patch"])
            if name == self.dinov3_teacher_name:
                patch_loss = self.dinov3_patch_gamma * patch_loss
            out[f"{name}_cls"] = cls_loss
            out[f"{name}_patch"] = patch_loss
            term = cls_loss + patch_loss
            total = term if total is None else total + term
        if total is None:
            total = torch.zeros((), dtype=torch.float32)
        out["loss"] = total
        return out
