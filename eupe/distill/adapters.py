# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Per-teacher adapter heads that project student tokens into each teacher's space.

Paper §4.1: a 2-layer MLP — Linear(no bias) -> LayerNorm -> GELU -> Linear(no bias).
Hidden dim 1536 in Stage 1, 3072 in Stage 2&3. Lineage: NVlabs/RADIO radio/adaptor_mlp.py
(MLP2), simplified to two layers. Adapters are trained with the student and DISCARDED at eval.
"""
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn


class AdapterHead(nn.Module):
    """Linear(no bias) -> LayerNorm -> GELU -> Linear(no bias).

    Args:
        in_dim: student feature dim (d_S).
        hidden_dim: 1536 (Stage 1) or 3072 (Stage 2&3).
        out_dim: target teacher feature dim (d_T).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # TODO: build fc1=Linear(in_dim,hidden_dim,bias=False), norm=LayerNorm(hidden_dim),
        # act=GELU, fc2=Linear(hidden_dim,out_dim,bias=False). See paper §4.1 / RADIO adaptor_mlp.MLP2.
        raise NotImplementedError("TODO: construct the 2-layer adapter MLP (paper §4.1)")

    def forward(self, x: Tensor) -> Tensor:
        """Map x[..., in_dim] -> [..., out_dim]. Works for cls [B,d] and patch [B,N,d]."""
        raise NotImplementedError("TODO: fc1 -> norm -> gelu -> fc2")


class AdapterHeadSet(nn.Module):
    """One (cls-head, patch-head) pair per teacher.

    Args:
        student_dim: d_S.
        teacher_specs: sequence of (teacher_name, teacher_dim).
        hidden_dim: adapter hidden dim for this stage.
    """

    def __init__(self, student_dim: int, teacher_specs: Sequence[Tuple[str, int]], hidden_dim: int):
        super().__init__()
        # TODO: nn.ModuleDict mapping name -> ModuleDict({"cls": AdapterHead, "patch": AdapterHead}).
        raise NotImplementedError("TODO: build per-teacher cls/patch adapter heads")

    def forward(self, student_cls: Tensor, student_patch: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """Return {teacher_name: {"cls": z_cls[B,d_T], "patch": z_patch[B,N_S,d_T]}}."""
        raise NotImplementedError("TODO: apply each teacher's cls/patch heads to the student tokens")
