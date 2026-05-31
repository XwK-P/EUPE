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
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eupe.distill import AdapterHeadSet, DistillationLoss, build_teachers
from eupe.distill.normalize import FeatureNormalizer, estimate_teacher_statistics

logger = logging.getLogger("eupe")


class DistillationMetaArch(nn.Module):
    """Build student + teachers + adapters + normalizers + loss from cfg.

    Args:
        cfg: merged OmegaConf with student/distill/optim/crops sections.
    """

    # Ported from refs/dinov3/dinov3/train/ssl_meta_arch.py:SSLMetaArch — divergence: DINO/iBOT/
    # Sinkhorn/KoLeo/Gram are all stripped; the objective is the RADIO-style DistillationLoss over
    # per-teacher adapter heads + frozen feature normalizers (paper §3). The EMA teacher is gone:
    # teachers here are externally-loaded frozen foundation models (eupe.distill.build_teachers).
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # build_model_from_cfg returns (student, teacher_ema, embed_dim); we keep ONLY the student
        # (no EMA target — distillation targets are the frozen external teachers).
        from eupe.models import build_model_from_cfg

        student, _teacher_ema, embed_dim = build_model_from_cfg(cfg)
        self.student = student
        self.embed_dim = embed_dim  # student feature dim d_S

        # Frozen foundation/proxy teachers (already .cuda().eval().requires_grad_(False)).
        teachers = build_teachers(cfg)
        # Register so .to()/state_dict()/train() recurse correctly; they stay frozen + eval.
        self.teachers = nn.ModuleDict(teachers)
        self.teachers.requires_grad_(False)

        # One (cls, patch) adapter pair per teacher: student_dim -> hidden_dim -> teacher_dim.
        teacher_specs = [(name, teacher.embed_dim) for name, teacher in teachers.items()]
        self.adapters = AdapterHeadSet(
            self.embed_dim,
            teacher_specs,
            cfg.distill.adapter_hidden_dim,
        )

        # Placeholder per-coordinate normalizers (identity mean=0/std=1) filled in by
        # init_normalizer() before training. Buffers only — no trainable params.
        self.normalizers = nn.ModuleDict(
            {
                name: nn.ModuleDict(
                    {
                        "cls": FeatureNormalizer(teacher.embed_dim),
                        "patch": FeatureNormalizer(teacher.embed_dim),
                    }
                )
                for name, teacher in teachers.items()
            }
        )

        # RADIO-style multi-teacher feature-matching loss; OmegaConf -> plain kwargs.
        loss_kwargs = dict(cfg.distill.loss) if cfg.distill.get("loss") is not None else {}
        self.loss = DistillationLoss(**loss_kwargs)

        logger.info(
            "DistillationMetaArch built: student embed_dim=%d, teachers=%s, adapter_hidden_dim=%d",
            self.embed_dim,
            list(teachers.keys()),
            cfg.distill.adapter_hidden_dim,
        )

    def init_normalizer(self, data_loader) -> None:
        """Run estimate_teacher_statistics(...) once and store frozen normalizers (paper §3.3)."""
        # Ported from refs/RADIO/radio/feature_normalizer.py (via eupe.distill.normalize) — divergence:
        # estimate streaming fp32 mean/std over `normalizer_warmup_iters` batches, then copy into the
        # already-registered FeatureNormalizer buffers (keeps them in this module's state_dict).
        n_iters = self.cfg.distill.get("normalizer_warmup_iters", 500)
        # Stage-3 pyramid: when crops.global_crops_size is a list, measure the (proxy) teacher stats
        # across the SAME pyramid scales it is run at during training, not only its native resolution.
        gcs = self.cfg.crops.get("global_crops_size", None) if "crops" in self.cfg else None
        pyramid_scales = (
            [int(s) for s in gcs]
            if isinstance(gcs, (list, tuple)) or (hasattr(gcs, "__iter__") and not isinstance(gcs, (str, bytes, int)))
            else None
        )
        logger.info("Estimating teacher statistics over %d iterations (pyramid_scales=%s)...", n_iters, pyramid_scales)
        estimated = estimate_teacher_statistics(
            self.teachers, data_loader, n_iters=n_iters, pyramid_scales=pyramid_scales
        )
        for name, by_token in estimated.items():
            for token_type, normalizer in by_token.items():
                target = self.normalizers[name][token_type]
                target.set_stats(normalizer.mean, normalizer.std)
        logger.info("Teacher statistics frozen for teachers=%s", list(estimated.keys()))

    @torch.no_grad()
    def get_teacher_outputs(
        self, images: Tensor, override_resolution: Optional[int] = None
    ) -> Dict[str, Dict[str, Tensor]]:
        """Forward each frozen teacher at its native resolution; return raw {name:{cls,patch}}.

        Resize `images` per teacher.native_resolution before the forward (no_grad). When
        `override_resolution` is given (Stage-3 multi-resolution: the teacher samples its own pyramid
        scale per iteration) it is used instead of the teacher's fixed native_resolution.
        """
        # Ported from refs/dinov3/dinov3/train/ssl_meta_arch.py:get_teacher_output — divergence:
        # no DINO/iBOT heads or Sinkhorn centering; each teacher is run once at its own native
        # resolution and returns raw {cls,patch}. Resize is bicubic (matches the loss-side interp).
        outputs: Dict[str, Dict[str, Tensor]] = {}
        for name, teacher in self.teachers.items():
            res = override_resolution if override_resolution is not None else teacher.native_resolution
            if images.shape[-1] != res or images.shape[-2] != res:
                teacher_images = F.interpolate(
                    images,
                    size=(res, res),
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                teacher_images = images
            out = teacher(teacher_images)
            outputs[name] = {"cls": out["cls"], "patch": out["patch"]}
        return outputs

    def compute_losses(self, student_cls: Tensor, student_patch: Tensor,
                       teacher_outputs: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Adapt student tokens, normalize teacher tokens, call DistillationLoss."""
        # Ported from refs/dinov3/dinov3/train/ssl_meta_arch.py:compute_losses — divergence: instead
        # of DINO/iBOT/KoLeo accumulation, project the student into each teacher space (adapters),
        # standardize the teacher targets (frozen normalizers), and feed both to DistillationLoss.
        adapted_student = self.adapters(student_cls, student_patch)

        teacher_normalized: Dict[str, Dict[str, Tensor]] = {}
        for name, tokens in teacher_outputs.items():
            normalizer = self.normalizers[name]
            teacher_normalized[name] = {
                "cls": normalizer["cls"](tokens["cls"]),
                "patch": normalizer["patch"](tokens["patch"]),
            }

        return self.loss(adapted_student, teacher_normalized)

    def backprop_loss(self, loss: Tensor) -> None:
        """Plain (unscaled) backward; grad clipping is owned by the train loop. See dinov3."""
        # Ported from refs/dinov3/dinov3/train/ssl_meta_arch.py:backprop_loss — divergence: bf16
        # autocast/FSDP keep the backward unscaled (no GradScaler), so this is a plain backward().
        # The cfg.optim.clip_grad clip is applied in eupe/train/train.py::do_train (matching dinov3,
        # which clips in the loop) over BOTH the FSDP2-sharded student (DTensor-aware global norm) and
        # the (non-FSDP) adapter heads — clipped per-unit, not as one joint norm (see _clip_gradients).
        loss.backward()

    def forward_backward(self, data, *, iteration: int = 0, **ignored) -> Dict[str, Tensor]:
        """One train step: student forward -> compute_losses -> backprop_loss; return log dict."""
        # Ported from refs/dinov3/dinov3/train/ssl_meta_arch.py:forward_backward — divergence: a
        # single global crop tensor (no local crops / masks); student forward_features -> cls/patch;
        # teachers run at their native resolutions; DistillationLoss replaces the DINO/iBOT terms.
        del iteration, ignored
        images = self._extract_images(data).cuda(non_blocking=True)

        # Student forward (trainable). forward_features returns the x_norm_* token dict.
        student_out = self.student.forward_features(images)
        student_cls = student_out["x_norm_clstoken"]
        student_patch = student_out["x_norm_patchtokens"]

        teacher_outputs = self.get_teacher_outputs(images)

        loss_dict = self.compute_losses(student_cls, student_patch, teacher_outputs)

        self.backprop_loss(loss_dict["loss"])

        return loss_dict

    @staticmethod
    def _extract_images(data) -> Tensor:
        """Pull the global-crop image tensor out of whatever the data loader yields."""
        # Mirror eupe.distill.normalize._extract_images so the trainer + normalizer agree on the
        # distillation batch schema (bare tensor / (images, ...) / dict with a known image key).
        if isinstance(data, Tensor):
            return data
        if isinstance(data, dict):
            for key in ("collated_global_crops", "images", "image", "img"):
                if key in data:
                    return data[key]
            for value in data.values():
                if isinstance(value, Tensor):
                    return value
            raise KeyError("forward_backward: no image tensor found in batch dict")
        if isinstance(data, (tuple, list)):
            return data[0]
        raise TypeError(f"forward_backward: unsupported batch type {type(data)!r}")

    def train(self):
        """Keep frozen teachers in eval() even when the meta-arch is set to train()."""
        # Ported from refs/dinov3/dinov3/train/ssl_meta_arch.py:train — keep targets deterministic.
        super().train()
        self.teachers.eval()
        return self
