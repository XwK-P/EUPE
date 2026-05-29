# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""MultiDistillationMetaArch: 1 frozen proxy -> M students on GPU rank-subgroups (Stage 2/3).

Subclass of DistillationMetaArch. The proxy teacher is computed ONCE on the global batch on all
ranks; its outputs are broadcast to each student's rank-subgroup (eupe/distributed.new_subgroups,
set up by eupe/configs/config.py::setup_multidistillation); the student local to this rank
backprops independently. Mirrors dinov3/train/multidist_meta_arch.py structure with the loss
swapped for DistillationLoss. crops.teacher_to_student_resolution_scale downsamples proxy crops
to the student resolution (e.g. Stage-3 multi-res).
"""
import logging
from typing import Dict

import torch
from torch import Tensor

import eupe.distributed as distributed
from eupe.train.distill_meta_arch import DistillationMetaArch

logger = logging.getLogger("eupe")


class MultiDistillationMetaArch(DistillationMetaArch):
    """Co-distill the whole student family from one proxy in a single job."""

    def broadcast_to_subgroups(self, x: Tensor, *, global_batch_size: int, over_dim: int = 0) -> Tensor:
        """Broadcast proxy outputs computed on the global batch to each student's subgroup slice.

        See dinov3 multidist_meta_arch.broadcast_to_subgroups + eupe.distributed primitives.
        """
        raise NotImplementedError("TODO: subgroup broadcast of teacher features")

    def get_teacher_output(self, images: Tensor, *, global_batch_size: int) -> Dict[str, Dict[str, Tensor]]:
        """Run the frozen proxy once on the global batch, then broadcast to this rank's subgroup."""
        raise NotImplementedError("TODO: proxy forward once + broadcast_to_subgroups")

    def forward_backward(self, data, *, iteration: int = 0, **ignored) -> Dict[str, Tensor]:
        """Shared-proxy step: get_teacher_output -> local student forward -> compute_losses -> backprop."""
        raise NotImplementedError("TODO: orchestrate shared-teacher multi-student step")
