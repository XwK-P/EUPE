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
import torch.nn.functional as F
from torch import Tensor

import eupe.distributed as distributed
from eupe.train.distill_meta_arch import DistillationMetaArch

logger = logging.getLogger("eupe")


class MultiDistillationMetaArch(DistillationMetaArch):
    """Co-distill the whole student family from one proxy in a single job."""

    # Ported from refs/dinov3/dinov3/train/multidist_meta_arch.py:MultiDistillationMetaArch — the
    # subgroup-broadcast plumbing is identical to dinov3; the only divergence is the objective: the
    # single frozen proxy + RADIO-style DistillationLoss (inherited from DistillationMetaArch)
    # replaces dinov3's EMA teacher with DINO/iBOT/Sinkhorn heads.

    def broadcast_to_subgroups(self, x: Tensor, *, global_batch_size: int, over_dim: int = 0) -> Tensor:
        """Broadcast proxy outputs computed on the global batch to each student's subgroup slice.

        Gathers ``x`` (computed identically on every world rank) across the full world, concatenates
        along ``over_dim``, trims to the true ``global_batch_size`` (drops zero-padding from uneven
        sharding), then hands this rank's process subgroup its contiguous chunk.
        """
        # Ported from refs/dinov3/dinov3/train/ssl_meta_arch.py:broadcast_to_subgroups — divergence:
        # use eupe.distributed.gather_all_tensors over the default (world) group instead of a raw
        # torch.distributed.all_gather, and short-circuit when distributed is not enabled so the
        # arch is importable / single-process runnable.
        subgroup_size = distributed.get_subgroup_size()
        if not distributed.is_enabled() or subgroup_size <= 1:
            # Single process (or a degenerate single-rank subgroup): every rank already holds the
            # full global batch, so just trim the padding and return it unchanged.
            return x.narrow(dim=over_dim, start=0, length=global_batch_size).clone()

        gathered = distributed.gather_all_tensors(x, group=distributed.get_default_process_group())
        catted = torch.cat(gathered, dim=over_dim)
        catted = catted.narrow(dim=over_dim, start=0, length=global_batch_size)
        return catted.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()

    @torch.no_grad()
    def get_teacher_output(self, images: Tensor, *, global_batch_size: int) -> Dict[str, Dict[str, Tensor]]:
        """Run the frozen proxy once on the global batch, then broadcast to this rank's subgroup."""
        # Ported from refs/dinov3/dinov3/train/multidist_meta_arch.py:get_teacher_output —
        # divergence: dinov3 runs DINO/iBOT heads + Sinkhorn centering and broadcasts the centered
        # logits; here the single proxy returns raw {cls,patch} (DistillationMetaArch.get_teacher_outputs
        # already forwards each teacher at its native resolution under no_grad), so we just broadcast
        # each cls/patch tensor along the batch dim (over_dim=0) to this rank's subgroup slice.
        teacher_outputs = self.get_teacher_outputs(images)
        subgroup_outputs: Dict[str, Dict[str, Tensor]] = {}
        for name, tokens in teacher_outputs.items():
            subgroup_outputs[name] = {
                "cls": self.broadcast_to_subgroups(
                    tokens["cls"], global_batch_size=global_batch_size, over_dim=0
                ),
                "patch": self.broadcast_to_subgroups(
                    tokens["patch"], global_batch_size=global_batch_size, over_dim=0
                ),
            }
        return subgroup_outputs

    def forward_backward(self, data, *, iteration: int = 0, **ignored) -> Dict[str, Tensor]:
        """Shared-proxy step: get_teacher_output -> local student forward -> compute_losses -> backprop."""
        # Ported from refs/dinov3/dinov3/train/multidist_meta_arch.py:forward_backward — divergence:
        # no global/local crops or masks; a single global-crop tensor is run through the proxy ONCE
        # (replicated on every world rank), the teacher output is broadcast to this rank's subgroup,
        # and the local student backprops against it via the inherited DistillationLoss.
        del iteration, ignored
        images = self._extract_images(data).cuda(non_blocking=True)
        global_batch_size = self._extract_global_batch_size(data, images)

        # Downsample the proxy crops to the student resolution (Stage-3 multi-res). The teacher still
        # runs once on the (down)sampled global batch on every rank before the subgroup broadcast.
        downsampling_factor = float(self.cfg.crops.get("teacher_to_student_resolution_scale", 1.0))
        if downsampling_factor != 1.0:
            images = F.interpolate(
                images,
                scale_factor=1.0 / downsampling_factor,
                mode="bilinear",
                antialias=True,
            )

        # Shared teacher: run the frozen proxy once on the global batch, then hand this rank's
        # subgroup its slice (all-gather -> cat -> narrow -> chunk).
        teacher_outputs = self.get_teacher_output(images, global_batch_size=global_batch_size)

        # Local student: each subgroup only owns its slice of the global batch.
        student_images = self.broadcast_to_subgroups(
            images, global_batch_size=global_batch_size, over_dim=0
        )
        student_out = self.student.forward_features(student_images)
        student_cls = student_out["x_norm_clstoken"]
        student_patch = student_out["x_norm_patchtokens"]

        loss_dict = self.compute_losses(student_cls, student_patch, teacher_outputs)

        self.backprop_loss(loss_dict["loss"])

        return loss_dict

    def _extract_global_batch_size(self, data, images: Tensor) -> int:
        """Resolve the true global batch size for the subgroup broadcast.

        Prefers an explicit ``global_batch_size`` carried in the data dict (dinov3 schema), then the
        ``multidistillation.global_batch_size`` config, finally falling back to the local batch (the
        single-process / non-distributed case where every rank already holds the whole batch).
        """
        if isinstance(data, dict) and "global_batch_size" in data:
            return int(data["global_batch_size"])
        multidist_cfg = self.cfg.get("multidistillation")
        if multidist_cfg is not None and multidist_cfg.get("global_batch_size") is not None:
            return int(multidist_cfg.global_batch_size)
        return int(images.shape[0])
