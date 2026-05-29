# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""DistillationMetaArch: N frozen teachers -> 1 trainable student.

Used directly for Stage 1 (3 foundation teachers -> ViT-G proxy) and as the per-student unit in
Stage 2/3 (1 proxy teacher -> 1 efficient student). Holds the student backbone, frozen teachers,
per-teacher adapter heads, frozen feature normalizers, and the DistillationLoss. Replaces
DINOv3's DINO/iBOT/Sinkhorn objective with RADIO-style feature matching (paper §3).
"""
import logging
from typing import Dict

import torch
from torch import Tensor, nn

from eupe.distill import AdapterHeadSet, DistillationLoss, build_teachers

logger = logging.getLogger("eupe")


class DistillationMetaArch(nn.Module):
    """Build student + teachers + adapters + normalizers + loss from cfg.

    Args:
        cfg: merged OmegaConf with student/distill/optim/crops sections.
    """

    def __init__(self, cfg):
        super().__init__()
        # TODO: build student (eupe.models.build_model_from_cfg, only student) on meta device;
        # build_teachers(cfg); AdapterHeadSet(student_dim, [(name, t.embed_dim)...], cfg.distill.adapter_hidden_dim);
        # placeholder normalizers (filled by init_normalizer); DistillationLoss(**cfg.distill.loss).
        raise NotImplementedError("TODO: assemble student/teachers/adapters/normalizer/loss")

    def init_normalizer(self, data_loader) -> None:
        """Run estimate_teacher_statistics(...) once and store frozen normalizers (paper §3.3)."""
        raise NotImplementedError("TODO: estimate + freeze teacher stats before training")

    def get_teacher_outputs(self, images: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """Forward each frozen teacher at its native resolution; return raw {name:{cls,patch}}.

        Resize `images` per teacher.native_resolution before the forward (no_grad).
        """
        raise NotImplementedError("TODO: per-teacher resize + frozen forward")

    def compute_losses(self, student_cls: Tensor, student_patch: Tensor,
                       teacher_outputs: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Adapt student tokens, normalize teacher tokens, call DistillationLoss."""
        raise NotImplementedError("TODO: adapters -> normalize teachers -> DistillationLoss")

    def backprop_loss(self, loss: Tensor) -> None:
        """Backward with the configured grad scaler / fp32 reduce. See dinov3 ssl_meta_arch."""
        raise NotImplementedError("TODO: scaled backward")

    def forward_backward(self, data, *, iteration: int = 0, **ignored) -> Dict[str, Tensor]:
        """One train step: student forward -> compute_losses -> backprop_loss; return log dict."""
        raise NotImplementedError("TODO: orchestrate one step")
