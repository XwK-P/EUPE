# Multi-Stage Multi-Teacher Distillation Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete, structurally-finished scaffold of EUPE's 3-stage multi-teacher distillation pipeline (the part the official repo stripped out) so a GPU-equipped team can fill in the stub bodies and run it.

**Architecture:** A DINOv3-style trainer (`eupe/train/`) carrying a RADIO-style feature-matching objective (`eupe/distill/`). `DistillationMetaArch` handles *N frozen teachers → 1 student* (Stage 1, proxy). `MultiDistillationMetaArch` subclasses it for *1 proxy → M students on GPU rank-subgroups* (Stage 2/3). FSDP/sharding in `eupe/fsdp/`; mixed LVD+ImageNet data in `eupe/data/`; concrete recipe-bearing configs in `eupe/configs/train/`.

**Tech Stack:** Python 3.11, PyTorch ≥ 2.7.1, FSDP, OmegaConf, submitit/SLURM. Reference repos: `facebookresearch/dinov3` (trainer/FSDP), `NVlabs/RADIO` (objective).

**IMPORTANT — deliverable policy (do not deviate):**
- Every Python function/method body is `raise NotImplementedError("TODO: …")` plus a docstring naming the exact paper §/equation or repo file to copy from. **Implement no real logic** — including loss/adapter/normalizer math.
- **Do NOT write tests.** Per-task verification is `python -m py_compile` (Python) or a YAML parse (configs). This is a deliberate decision by the repo owner.
- Config YAMLs are **concrete** (real batch sizes, LRs, iterations, rank ranges) with `<PATH/…>` placeholders for env-specific values.
- Every `.py` starts with the repo's FAIR license header (copy from any existing `eupe/*.py`). Use `logging.getLogger("eupe")` where a logger is needed.
- Reference spec: `docs/superpowers/specs/2026-05-29-multistage-multiteacher-distillation-design.md`. Recipe rationale: `../EUPE_Distillation_Reproduction_Report.md`.
- Work on branch `feat/multistage-distillation` (already created). Commit after each task. Do not push.

**License header to prepend to every `.py`:**
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.
```

---

## File Structure

| Path | Responsibility |
|---|---|
| `eupe/distill/__init__.py` | export `TeacherModel, build_teachers, AdapterHead, AdapterHeadSet, FeatureNormalizer, estimate_teacher_statistics, DistillationLoss` |
| `eupe/distill/teachers.py` | frozen teacher interface + PEcore/PElang/DINOv3/Proxy loaders + `build_teachers` |
| `eupe/distill/adapters.py` | 2-layer MLP adapter head + per-teacher head set |
| `eupe/distill/normalize.py` | frozen mean/std feature normalizer + 500-iter estimator |
| `eupe/distill/loss.py` | cosine(cls) + 0.9·cos/0.1·smoothL1(patch) summed over teachers |
| `eupe/train/__init__.py` | export the two meta-archs |
| `eupe/train/distill_meta_arch.py` | N teachers → 1 student engine (Stage 1) |
| `eupe/train/multidist_meta_arch.py` | 1 proxy → M students on rank-subgroups (Stage 2/3) |
| `eupe/train/param_groups.py` | AdamW param groups (layerwise decay, patch-embed mult) |
| `eupe/train/cosine_lr_scheduler.py` | cosine LR/WD/momentum schedules |
| `eupe/train/train.py` | training entry point + loop orchestration |
| `eupe/fsdp/__init__.py` | FSDP helper exports |
| `eupe/fsdp/ac_compile_parallelize.py` | FSDP + activation ckpt + compile |
| `eupe/data/distillation_loaders.py` | LVD+ImageNet mixed sampler + pyramid collate |
| `eupe/configs/train/teachers/{pecore_g,pelang_g,dinov3_hplus}.yaml` | teacher specs |
| `eupe/configs/train/proxy/vitg_p16.yaml` | proxy arch |
| `eupe/configs/train/students/{vitt,vits,vitb,convnext_tiny,convnext_small,convnext_base}_p16.yaml` | student arches |
| `eupe/configs/train/{stage1_multiteacher_proxy,stage2_multidistill,stage3_multidistill}.yaml` | stage orchestration |
| `DISTILLATION.md` | launch commands + fill-in checklist |

---

## Task 1: Package skeletons

**Files:**
- Create: `eupe/distill/__init__.py`
- Create: `eupe/train/__init__.py`
- Create: `eupe/fsdp/__init__.py`

- [ ] **Step 1: Create `eupe/distill/__init__.py`**

```python
# <FAIR header>
"""EUPE distillation objective: teachers, adapters, normalization, loss."""

from .adapters import AdapterHead, AdapterHeadSet
from .loss import DistillationLoss
from .normalize import FeatureNormalizer, estimate_teacher_statistics
from .teachers import TeacherModel, build_teachers

__all__ = [
    "TeacherModel",
    "build_teachers",
    "AdapterHead",
    "AdapterHeadSet",
    "FeatureNormalizer",
    "estimate_teacher_statistics",
    "DistillationLoss",
]
```

- [ ] **Step 2: Create `eupe/train/__init__.py`**

```python
# <FAIR header>
"""EUPE distillation trainer (DINOv3-style) for the multi-stage pipeline."""

from .distill_meta_arch import DistillationMetaArch
from .multidist_meta_arch import MultiDistillationMetaArch

__all__ = ["DistillationMetaArch", "MultiDistillationMetaArch"]
```

- [ ] **Step 3: Create `eupe/fsdp/__init__.py`**

```python
# <FAIR header>
"""FSDP sharding, activation checkpointing, and compile helpers."""

from .ac_compile_parallelize import (
    apply_activation_checkpointing,
    apply_compile,
    parallelize,
)

__all__ = ["parallelize", "apply_activation_checkpointing", "apply_compile"]
```

- [ ] **Step 4: Verify the directories are importable packages (syntax only)**

Run: `python -m py_compile eupe/distill/__init__.py eupe/train/__init__.py eupe/fsdp/__init__.py`
Expected: exit code 0, no output. (Imports will fail until later tasks create the modules — that's fine; `py_compile` only checks syntax of these files.)

- [ ] **Step 5: Commit**

```bash
git add eupe/distill/__init__.py eupe/train/__init__.py eupe/fsdp/__init__.py
git commit -m "scaffold: add distill/train/fsdp package inits"
```

---

## Task 2: Adapter heads — `eupe/distill/adapters.py`

**Files:**
- Create: `eupe/distill/adapters.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
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
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/distill/adapters.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/distill/adapters.py
git commit -m "scaffold: add adapter head stubs (paper 4.1)"
```

---

## Task 3: Feature normalizer — `eupe/distill/normalize.py`

**Files:**
- Create: `eupe/distill/normalize.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
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
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/distill/normalize.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/distill/normalize.py
git commit -m "scaffold: add feature normalizer stubs (paper 3.3/4.1)"
```

---

## Task 4: Distillation loss — `eupe/distill/loss.py`

**Files:**
- Create: `eupe/distill/loss.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
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
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/distill/loss.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/distill/loss.py
git commit -m "scaffold: add distillation loss stubs (paper 3.2, Eq.4-7)"
```

---

## Task 5: Teachers — `eupe/distill/teachers.py`

**Files:**
- Create: `eupe/distill/teachers.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
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
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/distill/teachers.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/distill/teachers.py
git commit -m "scaffold: add frozen teacher interface + loaders"
```

---

## Task 6: Optimizer param groups + LR scheduler

**Files:**
- Create: `eupe/train/param_groups.py`
- Create: `eupe/train/cosine_lr_scheduler.py`

- [ ] **Step 1: Write `eupe/train/param_groups.py`**

```python
# <FAIR header>
"""AdamW parameter groups for distillation (mirrors dinov3/train/param_groups.py).

Applies layer-wise LR decay across ViT blocks (optim.layerwise_decay), a patch-embed LR
multiplier (optim.patch_embed_lr_mult=0.2), and no weight decay on norms/biases/tokens.
"""
from typing import Dict, List

from torch import nn


def get_params_groups_with_decay(
    model: nn.Module,
    lr: float,
    wd: float,
    *,
    layerwise_decay: float = 1.0,
    patch_embed_lr_mult: float = 0.2,
) -> List[Dict]:
    """Return AdamW param-group dicts with per-parameter lr_mult and wd.

    Assign each param a layer index (patch_embed=0, block i = i+1, head=last); scale lr by
    layerwise_decay ** (last - idx); zero wd for 1-D params (norms/biases) and *_token params.
    See dinov3/train/param_groups.py.
    """
    raise NotImplementedError("TODO: build decayed param groups")


def fuse_params_groups(groups: List[Dict]) -> List[Dict]:
    """Merge param groups that share (lr_mult, wd) to reduce optimizer overhead."""
    raise NotImplementedError("TODO: fuse groups by (lr_mult, wd) key")
```

- [ ] **Step 2: Write `eupe/train/cosine_lr_scheduler.py`**

```python
# <FAIR header>
"""Cosine schedules for LR / weight-decay / teacher momentum (mirrors dinov3).

Supports schedules-v2 (set peak directly) to bypass the implicit sqrt_wrt_1024 LR scaling in
eupe/configs/config.py::apply_scaling_rules_to_cfg, so reproducers hit the paper's 2e-5 (Stage 2)
/ 1e-5 (Stage 3) regardless of GPU count.
"""
import numpy as np


class CosineScheduler:
    """Per-iteration schedule value via __getitem__(iteration).

    Args:
        base_value, final_value: endpoints.
        total_iters: schedule length.
        warmup_iters: linear warmup length.
        start_warmup_value: warmup start.
        freeze_iters: hold start_warmup_value for this many iters first.
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
    ):
        # TODO: precompute self.schedule as a numpy array of length total_iters
        # (freeze -> linear warmup -> cosine decay). See dinov3/train/cosine_lr_scheduler.py.
        raise NotImplementedError("TODO: precompute freeze+warmup+cosine schedule array")

    def __getitem__(self, it: int) -> float:
        raise NotImplementedError("TODO: return self.schedule[min(it, len-1)]")
```

- [ ] **Step 3: Verify**

Run: `python -m py_compile eupe/train/param_groups.py eupe/train/cosine_lr_scheduler.py`
Expected: exit code 0, no output.

- [ ] **Step 4: Commit**

```bash
git add eupe/train/param_groups.py eupe/train/cosine_lr_scheduler.py
git commit -m "scaffold: add param groups + cosine scheduler stubs"
```

---

## Task 7: Single-student engine — `eupe/train/distill_meta_arch.py`

**Files:**
- Create: `eupe/train/distill_meta_arch.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
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
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/train/distill_meta_arch.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/train/distill_meta_arch.py
git commit -m "scaffold: add single-student distillation meta-arch"
```

---

## Task 8: Co-distillation engine — `eupe/train/multidist_meta_arch.py`

**Files:**
- Create: `eupe/train/multidist_meta_arch.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
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
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/train/multidist_meta_arch.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/train/multidist_meta_arch.py
git commit -m "scaffold: add multi-student co-distillation meta-arch"
```

---

## Task 9: Training entry point — `eupe/train/train.py`

**Files:**
- Create: `eupe/train/train.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
"""Distillation training entry point.

Launched via:  python -m eupe.run.submit eupe/train/train.py --nodes N [--multi-distillation] \
                 --config-file eupe/configs/train/stageX_*.yaml --output-dir <OUT>
Builds the meta-arch on meta device, FSDP-wraps it, runs the 500-iter normalizer warmup, then
iterates the LVD+ImageNet sampler, applying cosine LR/WD/momentum schedules and periodic
checkpointing. Routes to MultiDistillationMetaArch when cfg.multidistillation.enabled.
"""
import argparse
import logging

from eupe.configs import setup_config, setup_multidistillation
from eupe.fsdp import parallelize
from eupe.run.init import job_context

logger = logging.getLogger("eupe")


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """--config-file, --output-dir, --multi-distillation, opts (mirrors dinov3/train/train.py)."""
    raise NotImplementedError("TODO: build the arg parser")


def build_optimizer(cfg, params_groups):
    """torch.optim.AdamW(betas=(adamw_beta1, adamw_beta2)). See dinov3."""
    raise NotImplementedError("TODO: build AdamW")


def build_schedulers(cfg) -> dict:
    """Return {"lr","wd","momentum","teacher_temp"} CosineSchedulers from cfg (schedules-v2 aware)."""
    raise NotImplementedError("TODO: build schedulers from cfg")


def apply_optim_scheduler(optimizer, schedulers, iteration: int) -> None:
    """Set per-group lr/wd from schedulers[iteration]."""
    raise NotImplementedError("TODO: write lr/wd into optimizer param groups")


def build_data_loader(cfg):
    """eupe.data.distillation_loaders.make_distillation_data_loader(cfg)."""
    raise NotImplementedError("TODO: build the LVD+ImageNet distillation loader")


def do_train(cfg, model) -> None:
    """Main loop: init_normalizer -> for it in range(max_iter): forward_backward; schedule; checkpoint."""
    raise NotImplementedError("TODO: implement the training loop")


def main(args=None) -> int:
    """Parse args, setup config (or setup_multidistillation), build+parallelize model, do_train."""
    raise NotImplementedError("TODO: wire setup_config/parallelize/do_train inside job_context")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/train/train.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/train/train.py
git commit -m "scaffold: add distillation training entry point"
```

---

## Task 10: FSDP wrapper — `eupe/fsdp/ac_compile_parallelize.py`

**Files:**
- Create: `eupe/fsdp/ac_compile_parallelize.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
"""FSDP sharding + activation checkpointing + torch.compile (mirrors dinov3/fsdp).

Reads cfg.compute_precision (param_dtype=bf16, reduce_dtype=fp32, sharding_strategy=SHARD_GRAD_OP)
and cfg.train.{checkpointing, checkpointing_full, compile}. Use FULL_SHARD for the 1.9B/7B proxy
if memory-bound.
"""
import logging

from torch import nn

logger = logging.getLogger("eupe")


def apply_activation_checkpointing(model: nn.Module, full: bool = False) -> nn.Module:
    """Wrap transformer blocks (or everything, if full) with activation checkpointing."""
    raise NotImplementedError("TODO: apply activation checkpointing to blocks")


def apply_compile(model: nn.Module) -> nn.Module:
    """torch.compile the per-block forward. See dinov3 ac_compile_parallelize."""
    raise NotImplementedError("TODO: torch.compile blocks")


def parallelize(model: nn.Module, cfg) -> nn.Module:
    """FSDP-wrap per cfg.compute_precision, optionally activation-ckpt + compile; return wrapped model.

    Build a MixedPrecision policy from param_dtype/reduce_dtype; choose ShardingStrategy from
    cfg.compute_precision.sharding_strategy; init on meta device then to_empty/cuda.
    """
    raise NotImplementedError("TODO: FSDP wrap with mixed precision + sharding strategy")
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/fsdp/ac_compile_parallelize.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/fsdp/ac_compile_parallelize.py
git commit -m "scaffold: add FSDP parallelize/ac/compile stubs"
```

---

## Task 11: Data loaders — `eupe/data/distillation_loaders.py`

**Files:**
- Create: `eupe/data/distillation_loaders.py`

- [ ] **Step 1: Write the stub module**

```python
# <FAIR header>
"""Distillation data: LVD-1689M + ImageNet-1k mix and the Stage-3 resolution pyramid (paper §3.4, §4.1).

Interleave homogeneous ImageNet-1k batches with heterogeneous LVD-1689M batches, P(ImageNet)=0.10.
Augmentations: random-resized crop, horizontal flip, color jitter, Gaussian blur, solarization
(Stage 2). Stage 3 adds an independent per-sample scale draw from {256,384,512}. Reuses
eupe/data/{loaders.py, samplers.py, transforms.py}.
"""
import logging
from typing import Iterable, List

logger = logging.getLogger("eupe")


class MixedSampler:
    """Yield ImageNet-1k batches with probability `imagenet_prob`, else LVD-1689M batches.

    Args:
        lvd_dataset, imagenet_dataset: the two sources.
        imagenet_prob: 0.10 (paper §3.4).
    """

    def __init__(self, lvd_dataset, imagenet_dataset, imagenet_prob: float = 0.10):
        raise NotImplementedError("TODO: store sources + Bernoulli(imagenet_prob) batch routing")

    def __iter__(self):
        raise NotImplementedError("TODO: yield homogeneous/heterogeneous batches by probability")


def build_pyramid_collate(scales: List[int]):
    """Return a collate_fn that resizes each sample to an independently sampled scale (Stage 3)."""
    raise NotImplementedError("TODO: per-sample random scale collate for {256,384,512}")


def make_distillation_data_loader(cfg) -> Iterable:
    """Build the mixed sampler + transforms + DataLoader from cfg (crops, batch_size_per_gpu, workers).

    If cfg.crops.global_crops_size is a list, attach build_pyramid_collate (Stage 3).
    """
    raise NotImplementedError("TODO: assemble datasets, MixedSampler, transforms, DataLoader")
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile eupe/data/distillation_loaders.py`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add eupe/data/distillation_loaders.py
git commit -m "scaffold: add LVD+ImageNet distillation data loader stubs"
```

---

## Task 12: Teacher + proxy configs

**Files:**
- Create: `eupe/configs/train/teachers/pecore_g.yaml`
- Create: `eupe/configs/train/teachers/pelang_g.yaml`
- Create: `eupe/configs/train/teachers/dinov3_hplus.yaml`
- Create: `eupe/configs/train/proxy/vitg_p16.yaml`

- [ ] **Step 1: Write `eupe/configs/train/teachers/pecore_g.yaml`**

```yaml
# PEcore-G teacher (image understanding / zero-shot). Load via perception_models.
name: pecore_g
loader: PECoreTeacher
checkpoint: <PATH/TO/PEcore-G.pt>   # facebookresearch/perception_models
native_resolution: 448
embed_dim: 1536
```

- [ ] **Step 2: Write `eupe/configs/train/teachers/pelang_g.yaml`**

```yaml
# PElang-G teacher (VLM / OCR). Load via perception_models. Crucial for OCR + general VLM.
name: pelang_g
loader: PELangTeacher
checkpoint: <PATH/TO/PElang-G.pt>   # facebookresearch/perception_models
native_resolution: 448
embed_dim: 1536
```

- [ ] **Step 3: Write `eupe/configs/train/teachers/dinov3_hplus.yaml`**

```yaml
# DINOv3-H+ teacher (dense prediction). Load via dinov3 hub / checkpoint.
name: dinov3_hplus
loader: DINOv3Teacher
checkpoint: <PATH/TO/dinov3_hplus.pth>   # facebookresearch/dinov3
native_resolution: 256
embed_dim: 1280
```

- [ ] **Step 4: Write `eupe/configs/train/proxy/vitg_p16.yaml`**

```yaml
# ViT-G proxy (~1.9B). Stage-1 student; Stage-2/3 teacher. Tune ffn/depth toward ~1.9B params.
student:
  arch: vit_giant2          # embed_dim 1536, depth 40, num_heads 24 (eupe/models/vision_transformer.py)
  patch_size: 16
  n_storage_tokens: 4       # 4 register tokens (paper §4.1)
  layerscale: 1.0e-05
  norm_layer: layernormbf16
  ffn_layer: mlp
  ffn_ratio: 4.0
  mask_k_bias: true
  pos_embed_type: rope
  pos_embed_rope_base: 100
  pos_embed_rope_normalize_coords: separate
  pos_embed_rope_rescale_coords: 2
  pos_embed_rope_dtype: fp32
```

- [ ] **Step 5: Verify all four parse as YAML**

Run:
```bash
python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('eupe/configs/train/teachers/*.yaml')+['eupe/configs/train/proxy/vitg_p16.yaml']]; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add eupe/configs/train/teachers eupe/configs/train/proxy
git commit -m "scaffold: add teacher + proxy configs"
```

---

## Task 13: Student configs

**Files:**
- Create: `eupe/configs/train/students/vitt_p16.yaml`
- Create: `eupe/configs/train/students/vits_p16.yaml`
- Create: `eupe/configs/train/students/vitb_p16.yaml`
- Create: `eupe/configs/train/students/convnext_tiny_p16.yaml`
- Create: `eupe/configs/train/students/convnext_small_p16.yaml`
- Create: `eupe/configs/train/students/convnext_base_p16.yaml`

- [ ] **Step 1: Write the three ViT student configs**

`vitt_p16.yaml`:
```yaml
# EUPE ViT-T/16 student (6M). Matches eupe/hub/backbones.py::eupe_vitt16.
student:
  arch: vit_small            # overridden by explicit dims below
  embed_dim: 192
  depth: 12
  num_heads: 3
  patch_size: 16
  n_storage_tokens: 4
  layerscale: 1.0e-05
  norm_layer: layernormbf16
  ffn_layer: mlp
  ffn_ratio: 4.0
  mask_k_bias: true
  drop_path_rate: 0.0
  pos_embed_type: rope
  pos_embed_rope_base: 100
  pos_embed_rope_normalize_coords: separate
  pos_embed_rope_rescale_coords: 2
  pos_embed_rope_dtype: fp32
```

`vits_p16.yaml`:
```yaml
# EUPE ViT-S/16 student (21M). Matches eupe/hub/backbones.py::eupe_vits16.
student:
  arch: vit_small
  embed_dim: 384
  depth: 12
  num_heads: 6
  patch_size: 16
  n_storage_tokens: 4
  layerscale: 1.0e-05
  norm_layer: layernormbf16
  ffn_layer: mlp
  ffn_ratio: 4.0
  mask_k_bias: true
  drop_path_rate: 0.0
  pos_embed_type: rope
  pos_embed_rope_base: 100
  pos_embed_rope_normalize_coords: separate
  pos_embed_rope_rescale_coords: 2
  pos_embed_rope_dtype: fp32
```

`vitb_p16.yaml`:
```yaml
# EUPE ViT-B/16 student (86M). Matches eupe/hub/backbones.py::eupe_vitb16.
student:
  arch: vit_base
  embed_dim: 768
  depth: 12
  num_heads: 12
  patch_size: 16
  n_storage_tokens: 4
  layerscale: 1.0e-05
  norm_layer: layernormbf16
  ffn_layer: mlp
  ffn_ratio: 4.0
  mask_k_bias: true
  drop_path_rate: 0.0
  pos_embed_type: rope
  pos_embed_rope_base: 100
  pos_embed_rope_normalize_coords: separate
  pos_embed_rope_rescale_coords: 2
  pos_embed_rope_dtype: fp32
```

- [ ] **Step 2: Write the three ConvNeXt student configs**

`convnext_tiny_p16.yaml`:
```yaml
# EUPE ConvNeXt-Tiny student (29M). Matches eupe/models/convnext.py::convnext_sizes["tiny"].
student:
  arch: convnext_tiny
  depths: [3, 3, 9, 3]
  dims: [96, 192, 384, 768]
  drop_path_rate: 0.0
  layer_scale_init_value: 1.0e-06
```

`convnext_small_p16.yaml`:
```yaml
# EUPE ConvNeXt-Small student (50M).
student:
  arch: convnext_small
  depths: [3, 3, 27, 3]
  dims: [96, 192, 384, 768]
  drop_path_rate: 0.0
  layer_scale_init_value: 1.0e-06
```

`convnext_base_p16.yaml`:
```yaml
# EUPE ConvNeXt-Base student (89M).
student:
  arch: convnext_base
  depths: [3, 3, 27, 3]
  dims: [128, 256, 512, 1024]
  drop_path_rate: 0.0
  layer_scale_init_value: 1.0e-06
```

- [ ] **Step 3: Verify all six parse as YAML**

Run:
```bash
python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('eupe/configs/train/students/*.yaml')]; print(len(glob.glob('eupe/configs/train/students/*.yaml')),'ok')"
```
Expected: prints `6 ok`.

- [ ] **Step 4: Commit**

```bash
git add eupe/configs/train/students
git commit -m "scaffold: add ViT + ConvNeXt student configs"
```

---

## Task 14: Stage orchestration configs

**Files:**
- Create: `eupe/configs/train/stage1_multiteacher_proxy.yaml`
- Create: `eupe/configs/train/stage2_multidistill.yaml`
- Create: `eupe/configs/train/stage3_multidistill.yaml`

- [ ] **Step 1: Write `eupe/configs/train/stage1_multiteacher_proxy.yaml`**

```yaml
# Stage 1: {PEcore-G, PElang-G, DINOv3-H+} -> ViT-G proxy. Single student (the proxy).
MODEL:
  META_ARCHITECTURE: DistillationMetaArch
compute_precision:
  param_dtype: bf16
  reduce_dtype: fp32
  sharding_strategy: SHARD_GRAD_OP   # use FULL_SHARD if memory-bound on the 1.9B proxy
distill:
  teachers:
    - {name: pecore_g,     config: teachers/pecore_g.yaml}
    - {name: pelang_g,     config: teachers/pelang_g.yaml}
    - {name: dinov3_hplus, config: teachers/dinov3_hplus.yaml}
  adapter_hidden_dim: 1536
  normalizer_warmup_iters: 500
  loss: {alpha: 0.9, beta: 0.1, dinov3_patch_gamma: 1.0}
student:                              # see proxy/vitg_p16.yaml; merge it in via opts at launch
  arch: vit_giant2
  patch_size: 16
  n_storage_tokens: 4
crops:
  global_crops_size: 256
optim:
  optimizer: adamw
  scaling_rule: sqrt_wrt_1024
  clip_grad: 3.0
  weight_decay: 0.04
  adamw_beta1: 0.9
  adamw_beta2: 0.999
train:
  dataset_path: <LVD1689M+IN1k>       # see eupe/data/distillation_loaders.py
  compile: true
  checkpointing: true
```

- [ ] **Step 2: Write `eupe/configs/train/stage2_multidistill.yaml`**

```yaml
# Stage 2: proxy -> efficient family, fixed 256, long schedule. Co-distill via rank-subgroups.
# Launch with: python -m eupe.run.submit eupe/train/train.py --multi-distillation --nodes 16 ...
MODEL:
  META_ARCHITECTURE: MultiDistillationMetaArch
compute_precision:
  param_dtype: bf16
  reduce_dtype: fp32
  sharding_strategy: SHARD_GRAD_OP
multidistillation:                    # rank-partition launcher (eupe/configs/config.py::setup_multidistillation)
  enabled: true
  global_batch_size: 8192
  students:
    - {name: vitt,           config_path: eupe/configs/train/students/vitt_p16.yaml,           ranks_range: [0, 16]}
    - {name: vits,           config_path: eupe/configs/train/students/vits_p16.yaml,           ranks_range: [16, 40]}
    - {name: vitb,           config_path: eupe/configs/train/students/vitb_p16.yaml,           ranks_range: [40, 88]}
    - {name: convnext_tiny,  config_path: eupe/configs/train/students/convnext_tiny_p16.yaml,  ranks_range: [88, 104]}
    - {name: convnext_small, config_path: eupe/configs/train/students/convnext_small_p16.yaml, ranks_range: [104, 128]}
    - {name: convnext_base,  config_path: eupe/configs/train/students/convnext_base_p16.yaml,  ranks_range: [128, 168]}
distill:
  teachers:
    - {name: proxy, config: eupe/configs/train/proxy/vitg_p16.yaml, checkpoint: <STAGE1_PROXY_CKPT.pth>}
  adapter_hidden_dim: 3072
  normalizer_warmup_iters: 500
  loss: {alpha: 0.9, beta: 0.1, dinov3_patch_gamma: 1.0}
crops:
  global_crops_size: 256
optim:
  optimizer: adamw
  weight_decay: 1.0e-04
  adamw_beta1: 0.9
  adamw_beta2: 0.999
schedules:                            # schedules-v2: set peak LR directly (target paper's 2e-5)
  lr: {start: 0.0, peak: 2.0e-05, end: 0.0, warmup_epochs: 0, cosine: true}
train:
  dataset_path: <LVD1689M+IN1k>
  OFFICIAL_EPOCH_LENGTH: 1250         # 390000 iters total -> set optim.epochs = 312
  compile: true
  checkpointing: true
optim_total_iters: 390000             # documented target; map to epochs via OFFICIAL_EPOCH_LENGTH
```

- [ ] **Step 3: Write `eupe/configs/train/stage3_multidistill.yaml`**

```yaml
# Stage 3: proxy -> family, multi-resolution pyramid {256,384,512}, short schedule.
# Students init from Stage-2 checkpoints (set student.pretrained_weights per student config).
MODEL:
  META_ARCHITECTURE: MultiDistillationMetaArch
compute_precision:
  param_dtype: bf16
  reduce_dtype: fp32
  sharding_strategy: SHARD_GRAD_OP
multidistillation:
  enabled: true
  global_batch_size: 4096
  students:
    - {name: vitt,           config_path: eupe/configs/train/students/vitt_p16.yaml,           ranks_range: [0, 16]}
    - {name: vits,           config_path: eupe/configs/train/students/vits_p16.yaml,           ranks_range: [16, 40]}
    - {name: vitb,           config_path: eupe/configs/train/students/vitb_p16.yaml,           ranks_range: [40, 88]}
    - {name: convnext_tiny,  config_path: eupe/configs/train/students/convnext_tiny_p16.yaml,  ranks_range: [88, 104]}
    - {name: convnext_small, config_path: eupe/configs/train/students/convnext_small_p16.yaml, ranks_range: [104, 128]}
    - {name: convnext_base,  config_path: eupe/configs/train/students/convnext_base_p16.yaml,  ranks_range: [128, 168]}
distill:
  teachers:
    - {name: proxy, config: eupe/configs/train/proxy/vitg_p16.yaml, checkpoint: <STAGE1_PROXY_CKPT.pth>}
  adapter_hidden_dim: 3072
  normalizer_warmup_iters: 500
  loss: {alpha: 0.9, beta: 0.1, dinov3_patch_gamma: 1.0}
crops:
  global_crops_size: [256, 384, 512]  # pyramid; teacher & student sample independently
  teacher_to_student_resolution_scale: 1.0
optim:
  optimizer: adamw
  weight_decay: 1.0e-04
  adamw_beta1: 0.9
  adamw_beta2: 0.999
schedules:
  lr: {start: 0.0, peak: 1.0e-05, end: 0.0, warmup_epochs: 0, cosine: true}
train:
  dataset_path: <LVD1689M+IN1k>
  OFFICIAL_EPOCH_LENGTH: 1250         # 100000 iters total -> set optim.epochs = 80
  compile: true
  checkpointing: true
optim_total_iters: 100000
```

- [ ] **Step 4: Verify all three parse as YAML**

Run:
```bash
python -c "import yaml; [yaml.safe_load(open('eupe/configs/train/'+f)) for f in ['stage1_multiteacher_proxy.yaml','stage2_multidistill.yaml','stage3_multidistill.yaml']]; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add eupe/configs/train/stage1_multiteacher_proxy.yaml eupe/configs/train/stage2_multidistill.yaml eupe/configs/train/stage3_multidistill.yaml
git commit -m "scaffold: add stage 1/2/3 orchestration configs"
```

---

## Task 15: `DISTILLATION.md` + final verification

**Files:**
- Create: `DISTILLATION.md`

- [ ] **Step 1: Write `DISTILLATION.md`**

````markdown
# EUPE Multi-Stage Distillation

Scaffold for reproducing EUPE's scaling-up → scaling-down distillation. See the design spec
(`docs/superpowers/specs/2026-05-29-multistage-multiteacher-distillation-design.md`) and the
engineering report (`../EUPE_Distillation_Reproduction_Report.md`) for full rationale.

> **Status: scaffold.** All Python bodies raise `NotImplementedError`; fill them in following the
> docstring references before running. Configs are concrete (paths are `<PLACEHOLDERS>`).

## Pipeline
1. **Stage 1** — distill PEcore-G + PElang-G + DINOv3-H+ into a ~1.9B ViT-G proxy.
2. **Stage 2** — distill the frozen proxy into the efficient family @256, bs 8192, lr 2e-5, 390k iters.
3. **Stage 3** — multi-resolution finetune @{256,384,512}, bs 4096, lr 1e-5, 100k iters (init from Stage 2).

## Launch
```bash
# Stage 1 (single student = proxy)
python -m eupe.run.submit eupe/train/train.py --nodes 32 --ngpus 8 \
  --config-file eupe/configs/train/stage1_multiteacher_proxy.yaml --output-dir <OUT>

# Stage 2 / 3 (co-distill the family; --multi-distillation enables rank-subgroups)
python -m eupe.run.submit eupe/train/train.py --nodes 16 --ngpus 8 --multi-distillation \
  --config-file eupe/configs/train/stage2_multidistill.yaml --output-dir <OUT>
```

## Fill-in checklist (each maps to one `NotImplementedError`)
- `eupe/distill/adapters.py` — 2-layer MLP (paper §4.1)
- `eupe/distill/normalize.py` — frozen mean/std + 500-iter estimator (paper §3.3)
- `eupe/distill/loss.py` — cosine + 0.9·cos/0.1·smoothL1 (paper §3.2, Eq. 4-7)
- `eupe/distill/teachers.py` — load PEcore-G/PElang-G (perception_models), DINOv3-H+ (dinov3), proxy
- `eupe/train/{param_groups,cosine_lr_scheduler}.py` — optimizer groups + schedules (dinov3)
- `eupe/train/{distill,multidist}_meta_arch.py` — step orchestration + subgroup broadcast (dinov3 multidist_meta_arch.py)
- `eupe/train/train.py` — loop + checkpointing
- `eupe/fsdp/ac_compile_parallelize.py` — FSDP(SHARD_GRAD_OP) + ac + compile (dinov3 fsdp)
- `eupe/data/distillation_loaders.py` — LVD+IN1k mix (p=0.10) + pyramid collate

## Validation milestones
- After Stage 1: reproduce proxy numbers (report Table 4).
- After Stage 2: "Stage 1&2" column (report Table 2).
- After Stage 3: final EUPE numbers (report Table 1), then evaluate with `eupe/eval/*`.

## LR scaling note
`eupe/configs/config.py` applies `sqrt_wrt_1024` to `optim.lr`. The stage configs use schedules-v2
(`schedules.lr.peak`) to set the peak LR directly (2e-5 / 1e-5) regardless of GPU count.
````

- [ ] **Step 2: Verify the whole scaffold compiles**

Run:
```bash
python -m py_compile $(git ls-files --others --cached --exclude-standard 'eupe/**/*.py' | tr '\n' ' ') && echo "all py ok"
```
Expected: prints `all py ok` (every new `.py` is syntactically valid).

- [ ] **Step 3: Verify every config parses**

Run:
```bash
python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('eupe/configs/train/**/*.yaml', recursive=True)]; print('configs ok')"
```
Expected: prints `configs ok`.

- [ ] **Step 4: Commit**

```bash
git add DISTILLATION.md
git commit -m "scaffold: add DISTILLATION.md launch + fill-in guide"
```

---

## Self-Review (completed by plan author)

**1. Spec coverage:** All 28 files in the spec's §6 manifest map to Tasks 1–15 (distill ×5 → T1-5; train ×6 → T1,6,7,8,9; fsdp ×2 → T1,10; data ×1 → T11; configs ×13 → T12,13,14; DISTILLATION.md → T15). The `distill:` config block (spec §4) appears in the Stage configs (T14). The two-engine architecture (spec §2) is T7/T8. ✓

**2. Placeholder scan:** No "TBD/implement later/handle edge cases". The `NotImplementedError`/`<PATH>` tokens are the *intended deliverable* per the agreed policy, not plan placeholders; each is paired with a concrete reference. ✓

**3. Type/name consistency across tasks:** `AdapterHeadSet`, `FeatureNormalizer`, `estimate_teacher_statistics`, `DistillationLoss`, `TeacherModel`, `build_teachers` (defined T2-5, exported T1, consumed T7); `DistillationMetaArch` (T7) subclassed by `MultiDistillationMetaArch` (T8); `parallelize` (T10) exported T1, used T9; `make_distillation_data_loader` (T11) used T9. Names match across tasks. ✓

**Fix applied:** T13 Step 1 now shows the full `vits_p16.yaml` and `vitb_p16.yaml` blocks (no "repeat the block" references), so every config step is fully mechanical and order-independent.
