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
        # Ported from refs/RADIO/radio/adaptor_mlp.py:MLP2 (num_inner=0) - biases removed per paper §4.1
        # MLP2 structure is fc1 -> (LayerNorm, GELU, Linear). With num_inner=0 the residual blocks
        # vanish, collapsing to fc1 -> norm -> act -> fc2; all Linear biases dropped (paper §4.1).
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Map x[..., in_dim] -> [..., out_dim]. Works for cls [B,d] and patch [B,N,d]."""
        # LayerNorm/GELU/Linear all act on the trailing feature dim, so leading dims
        # (the optional token axis) broadcast for free.
        return self.fc2(self.act(self.norm(self.fc1(x))))


class AdapterHeadSet(nn.Module):
    """One (cls-head, patch-head) pair per teacher.

    Args:
        student_dim: d_S.
        teacher_specs: sequence of (teacher_name, teacher_dim).
        hidden_dim: adapter hidden dim for this stage.
    """

    def __init__(self, student_dim: int, teacher_specs: Sequence[Tuple[str, int]], hidden_dim: int):
        super().__init__()
        # One independent (cls, patch) adapter pair per teacher; student_dim -> hidden_dim -> teacher_dim.
        self.heads = nn.ModuleDict(
            {
                name: nn.ModuleDict(
                    {
                        "cls": AdapterHead(student_dim, hidden_dim, teacher_dim),
                        "patch": AdapterHead(student_dim, hidden_dim, teacher_dim),
                    }
                )
                for name, teacher_dim in teacher_specs
            }
        )

    def forward(self, student_cls: Tensor, student_patch: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """Return {teacher_name: {"cls": z_cls[B,d_T], "patch": z_patch[B,N_S,d_T]}}."""
        return {
            name: {
                "cls": head["cls"](student_cls),
                "patch": head["patch"](student_patch),
            }
            for name, head in self.heads.items()
        }
