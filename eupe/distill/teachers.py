# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Frozen teacher models for distillation.

Stage 1 teachers: PEcore-G (1.9B, 448), PElang-G (1.7B, 448) from facebookresearch/perception_models;
DINOv3-H+ (840M, 256) from facebookresearch/dinov3. Stage 2/3 teacher: the Stage-1 proxy (ViT-G).
Every teacher is frozen and exposes a class token + patch tokens at its native resolution.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor, nn

logger = logging.getLogger("eupe")


class TeacherModel(ABC, nn.Module):
    """Frozen teacher interface.

    Attributes:
        native_resolution: input square size the teacher expects (448 PE, 256 DINOv3-H+/proxy).
        embed_dim: feature dim of cls/patch tokens.
    """

    native_resolution: int
    embed_dim: int

    @abstractmethod
    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        """Return {"cls": [B, embed_dim], "patch": [B, N, embed_dim]} under no_grad."""
        raise NotImplementedError


class PECoreTeacher(TeacherModel):
    """PEcore-G image-understanding teacher. TODO: load via perception_models (open_clip-style)."""

    def __init__(self, checkpoint: str, native_resolution: int = 448):
        super().__init__()
        raise NotImplementedError("TODO: load PEcore-G, freeze, set embed_dim/native_resolution")

    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError("TODO: return frozen cls/patch tokens")


class PELangTeacher(TeacherModel):
    """PElang-G VLM/OCR teacher. TODO: load via perception_models."""

    def __init__(self, checkpoint: str, native_resolution: int = 448):
        super().__init__()
        raise NotImplementedError("TODO: load PElang-G, freeze")

    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError("TODO: return frozen cls/patch tokens")


class DINOv3Teacher(TeacherModel):
    """DINOv3-H+ dense-prediction teacher. TODO: load via dinov3 hub / checkpoint."""

    def __init__(self, checkpoint: str, native_resolution: int = 256):
        super().__init__()
        raise NotImplementedError("TODO: load DINOv3-H+, freeze")

    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError("TODO: return frozen cls/patch tokens (x_norm_clstoken / x_norm_patchtokens)")


class ProxyTeacher(TeacherModel):
    """Stage-1 proxy (ViT-G) used as the single teacher in Stage 2/3.

    TODO: build eupe DinoVisionTransformer (vit_giant2) from `config`, load `checkpoint`
    (handle 'teacher.'-prefixed keys, see eupe/models/__init__.build_model_for_eval), freeze.
    """

    def __init__(self, config: str, checkpoint: str, native_resolution: int = 256):
        super().__init__()
        raise NotImplementedError("TODO: build vit_giant2 proxy, load checkpoint, freeze")

    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError("TODO: return frozen cls/patch tokens from forward_features")


_TEACHER_REGISTRY = {
    "pecore_g": PECoreTeacher,
    "pelang_g": PELangTeacher,
    "dinov3_hplus": DINOv3Teacher,
    "proxy": ProxyTeacher,
}


def build_teachers(cfg) -> Dict[str, TeacherModel]:
    """Instantiate teachers from cfg.distill.teachers (list of {name, config?, checkpoint?, ...}).

    A list of >1 teacher ⇒ Stage 1 (multi-teacher). A single 'proxy' entry ⇒ Stage 2/3.
    Each teacher is moved to cuda, set to eval(), and has requires_grad_(False).
    """
    raise NotImplementedError("TODO: read cfg.distill.teachers, look up _TEACHER_REGISTRY, freeze each")
