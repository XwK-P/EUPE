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
import os
from abc import ABC, abstractmethod
from typing import Dict

import torch
from omegaconf import OmegaConf
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
        ...


# Ported from refs/perception_models/core/vision_encoder/config.py::PE_VISION_CONFIG
# - Map the EUPE teacher `name` onto a perception_models vision-config key.
_PE_CONFIG_FOR_NAME = {
    "pecore_g": "PE-Core-G14-448",
    "pelang_g": "PE-Lang-G14-448",
}


class _PEVisionTeacher(TeacherModel):
    """Shared loader for the perception_models PE vision encoders (PE-core / PE-lang).

    Both PEcore-G and PElang-G are the same `VisionTransformer` architecture (width 1536, G14)
    differing only by config key + checkpoint, so the load/forward logic is shared here.
    """

    # Ported from refs/perception_models/core/vision_encoder/pe.py::VisionTransformer
    # - Build via from_config(name) then load_ckpt(checkpoint); cls = model._pool(forward_features(img)),
    #   patch = forward_features(img)[:, int(use_cls_token):]. We call forward_features once with norm=True
    #   (its `forward` applies ln_post before pooling) and reuse the result for both cls and patch.
    def __init__(self, config_name: str, checkpoint: str, native_resolution: int = 448):
        super().__init__()
        # LAZY import so this module imports without perception_models installed.
        from core.vision_encoder.pe import VisionTransformer

        self.native_resolution = native_resolution
        model = VisionTransformer.from_config(config_name, pretrained=False)
        if checkpoint and not checkpoint.startswith("<"):
            model.load_ckpt(checkpoint, verbose=False)
        else:
            logger.warning("No checkpoint provided for PE teacher %s; using randomly-initialized weights", config_name)
        self.model = model
        # Token feature dim is the transformer width (the projection head is *not* applied in forward_features).
        self.embed_dim = model.width

    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        # Apply the post-norm so pooled cls + patch tokens are taken from the normalized features,
        # matching VisionTransformer.forward which runs forward_features(norm=True) before _pool.
        tokens = self.model.forward_features(img, norm=True)
        has_cls = bool(getattr(self.model, "use_cls_token", False))
        patch = tokens[:, 1:] if has_cls else tokens
        # _pool returns [B, d] for pool_type in {attn, avg, tok} (PE-Core uses attn). PE-Lang sets
        # pool_type="none", so _pool returns the FULL [B, N, d] sequence — fall back to a mean summary
        # (or the cls token when one exists) so "cls" is always a [B, d] vector for the cls adapter/loss.
        # FIDELITY NOTE: PE-Lang has no native class token, so its "cls" distillation target is the MEAN
        # of its own patch tokens (a defensible reconstruction — the paper does not specify how to take a
        # class token from a pool_type="none" encoder; this mirrors how PE-Core attn-pools the same
        # normed sequence). It makes PE-Lang's cls loss partially redundant with its patch loss.
        pooled = self.model._pool(tokens)
        if pooled.ndim == 2:
            cls = pooled
        elif has_cls:
            cls = tokens[:, 0]
        else:
            cls = tokens.mean(dim=1)
        return {"cls": cls, "patch": patch}


class PECoreTeacher(_PEVisionTeacher):
    """PEcore-G image-understanding teacher (facebookresearch/perception_models)."""

    def __init__(self, checkpoint: str, native_resolution: int = 448, config_name: str | None = None):
        # The perception_models vision-config key may be supplied via the teacher YAML (`pe_config`);
        # otherwise default to the canonical PE-Core-G14-448 (width 1536, pool_type=attn) for this loader.
        super().__init__(config_name or _PE_CONFIG_FOR_NAME["pecore_g"], checkpoint, native_resolution)


class PELangTeacher(_PEVisionTeacher):
    """PElang-G VLM/OCR teacher (facebookresearch/perception_models)."""

    def __init__(self, checkpoint: str, native_resolution: int = 448, config_name: str | None = None):
        # Config key via the teacher YAML (`pe_config`) or the default PE-Lang-G14-448 (pool_type=none,
        # so _PEVisionTeacher.forward mean-pools the cls). See PECoreTeacher.
        super().__init__(config_name or _PE_CONFIG_FOR_NAME["pelang_g"], checkpoint, native_resolution)


class DINOv3Teacher(TeacherModel):
    """DINOv3-H+ dense-prediction teacher (facebookresearch/dinov3)."""

    # Ported from refs/dinov3/dinov3/models/vision_transformer.py::DinoVisionTransformer.forward_features
    # - Returns the dict {"x_norm_clstoken", "x_norm_patchtokens", ...}; we surface cls/patch from it.
    #   Loaded through the hub entrypoint `dinov3_vith16plus` (DINOv3-H+, 840M, embed_dim 1280).
    def __init__(self, checkpoint: str, native_resolution: int = 256,
                 hub_entrypoint: str = "dinov3_vith16plus", embed_dim: int = 1280):
        super().__init__()
        self.native_resolution = native_resolution
        # embed_dim + hub entrypoint default to DINOv3-H+ (840M, 1280) but are overridable from the
        # teacher YAML (`hub_entrypoint`, `embed_dim`) so other DINOv3 variants can be swapped in.
        self.embed_dim = embed_dim
        has_ckpt = bool(checkpoint) and not checkpoint.startswith("<")
        try:
            # LAZY import: torch.hub.load pulls in the dinov3 package only when a teacher is built.
            self.model = torch.hub.load(
                "facebookresearch/dinov3",
                hub_entrypoint,
                source="github",
                pretrained=has_ckpt,
                weights=checkpoint if has_ckpt else None,
            )
        except Exception:
            # Fall back to a locally cloned dinov3 repo (offline / no network). DINOV3_LOCATION points
            # at the dinov3 source tree; mirrors the upstream local-hub recipe.
            dinov3_location = os.environ.get("DINOV3_LOCATION", "facebookresearch/dinov3")
            self.model = torch.hub.load(
                dinov3_location,
                hub_entrypoint,
                source="local" if os.path.isdir(dinov3_location) else "github",
                pretrained=has_ckpt,
                weights=checkpoint if has_ckpt else None,
            )

    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        out = self.model.forward_features(img)
        return {"cls": out["x_norm_clstoken"], "patch": out["x_norm_patchtokens"]}


class ProxyTeacher(TeacherModel):
    """Stage-1 proxy (ViT-G) used as the single teacher in Stage 2/3."""

    # Ported from eupe/models/__init__.py::build_model_for_eval / build_model
    # - Build the proxy backbone via eupe.models.build_model(student_cfg, only_teacher=True, img_size=res),
    #   then load the 'teacher.'-prefixed keys out of the Stage-1 checkpoint. forward via the
    #   DinoVisionTransformer.forward_features dict (same x_norm_* keys as DINOv3Teacher).
    def __init__(self, config: str, checkpoint: str, native_resolution: int = 256):
        super().__init__()
        from eupe.configs.config import get_default_config
        from eupe.models import build_model, extract_backbone_state_dict

        self.native_resolution = native_resolution

        proxy_cfg = OmegaConf.load(config)
        student_cfg = proxy_cfg.student if "student" in proxy_cfg else proxy_cfg
        # Merge onto the default student block FIRST. The proxy YAML only carries the explicit dims +
        # a rope subset; build_model unconditionally dereferences base-only keys (qkv_bias, ffn_bias,
        # proj_bias, the rope min/max/shift/jitter periods, the untie-norm flags, ...). Without this
        # merge, `OmegaConf.load(vitg_p16.yaml)` lacks those keys and build_model raises
        # ConfigAttributeError, crashing every Stage-2/3 proxy-teacher build. (The Stage-1 student path
        # gets these via setup_config's default merge; ProxyTeacher bypasses that, so it must merge here.)
        student_cfg = OmegaConf.merge(get_default_config().student, student_cfg)
        model, embed_dim = build_model(student_cfg, only_teacher=True, img_size=native_resolution)
        self.embed_dim = embed_dim

        if checkpoint and not checkpoint.startswith("<"):
            model.to_empty(device="cuda")
            # Handles the {"teacher": {"teacher.<k>": v}} training payload and flat/plain layouts.
            state_dict = extract_backbone_state_dict(torch.load(checkpoint, map_location="cpu"))
            model.load_state_dict(state_dict, strict=True)
        else:
            logger.warning("No checkpoint provided for proxy teacher; initializing weights")
            model.init_weights()
        self.model = model

    def forward(self, img: Tensor) -> Dict[str, Tensor]:
        out = self.model.forward_features(img)
        return {"cls": out["x_norm_clstoken"], "patch": out["x_norm_patchtokens"]}


_TEACHER_REGISTRY = {
    "pecore_g": PECoreTeacher,
    "pelang_g": PELangTeacher,
    "dinov3_hplus": DINOv3Teacher,
    "proxy": ProxyTeacher,
}

# A teacher YAML's `loader:` field may name either a short registry key (e.g. "pecore_g") or the
# loader class itself (e.g. "PECoreTeacher", as the shipped teacher configs do).
_LOADER_BY_CLASSNAME = {cls.__name__: cls for cls in set(_TEACHER_REGISTRY.values())}


def _resolve_teacher_entry(entry):
    """Merge an inline teacher entry with its referenced teacher config file (if any).

    Returns a flat OmegaConf-like mapping with keys: name, loader, checkpoint, native_resolution, ...
    The inline entry (from cfg.distill.teachers) takes precedence over the referenced file.
    """
    file_cfg = OmegaConf.create({})
    config_path = entry.get("config", None)
    if config_path is not None:
        # Resolve relative teacher-config paths against the eupe configs/train directory if needed.
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "train")
            candidate = os.path.join(base, config_path)
            if os.path.exists(candidate):
                config_path = candidate
        if os.path.exists(config_path):
            file_cfg = OmegaConf.load(config_path)
    merged = OmegaConf.merge(file_cfg, entry)
    return merged, config_path


def build_teachers(cfg) -> Dict[str, TeacherModel]:
    """Instantiate teachers from cfg.distill.teachers (list of {name, config?, checkpoint?, ...}).

    A list of >1 teacher ⇒ Stage 1 (multi-teacher). A single 'proxy' entry ⇒ Stage 2/3.
    Each teacher is moved to cuda, set to eval(), and has requires_grad_(False).
    """
    teachers: Dict[str, TeacherModel] = {}
    for entry in cfg.distill.teachers:
        merged, config_path = _resolve_teacher_entry(entry)
        name = merged["name"]
        # Prefer an explicit `loader` field (short registry key OR class name); else dispatch by name.
        loader_key = merged.get("loader", None)
        loader_cls = None
        if loader_key is not None:
            loader_cls = _TEACHER_REGISTRY.get(loader_key) or _LOADER_BY_CLASSNAME.get(loader_key)
        if loader_cls is None and name in _TEACHER_REGISTRY:
            loader_cls = _TEACHER_REGISTRY[name]
        if loader_cls is None:
            raise KeyError(f"No teacher loader registered for entry name={name!r} loader={loader_key!r}")

        checkpoint = merged.get("checkpoint", None)
        native_resolution = merged.get("native_resolution", None)

        if loader_cls is ProxyTeacher:
            # Proxy teacher needs the ViT-G student config (the merged path or the inline `config`).
            proxy_config = config_path if config_path is not None else merged.get("config")
            kwargs = {"config": proxy_config, "checkpoint": checkpoint}
            if native_resolution is not None:
                kwargs["native_resolution"] = native_resolution
            teacher = loader_cls(**kwargs)
        else:
            kwargs = {"checkpoint": checkpoint}
            if native_resolution is not None:
                kwargs["native_resolution"] = native_resolution
            # Config-driven external identifiers (let the teacher YAML supply what was previously
            # inferred from loader identity): PE config key, DINOv3 hub entrypoint + embed_dim.
            if loader_cls in (PECoreTeacher, PELangTeacher) and merged.get("pe_config") is not None:
                kwargs["config_name"] = merged.get("pe_config")
            if loader_cls is DINOv3Teacher:
                if merged.get("hub_entrypoint") is not None:
                    kwargs["hub_entrypoint"] = merged.get("hub_entrypoint")
                if merged.get("embed_dim") is not None:
                    kwargs["embed_dim"] = int(merged.get("embed_dim"))
            teacher = loader_cls(**kwargs)

        teacher = teacher.cuda().eval().requires_grad_(False)
        teachers[name] = teacher
        logger.info(
            "Built teacher %r (%s): embed_dim=%d native_resolution=%d",
            name,
            loader_cls.__name__,
            teacher.embed_dim,
            teacher.native_resolution,
        )
    return teachers
