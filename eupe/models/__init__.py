# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

import logging
from pathlib import Path

from typing import Sequence, Union

import torch
import torch.nn as nn

from . import vision_transformer as vits

logger = logging.getLogger("eupe")


def build_model(args, only_teacher=False, img_size=224, device=None):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            pos_embed_rope_base=args.pos_embed_rope_base,
            pos_embed_rope_min_period=args.pos_embed_rope_min_period,
            pos_embed_rope_max_period=args.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
            # Thread the rope dtype so the YAML (every ViT config sets fp32) is authoritative; otherwise
            # build_model fell back to DinoVisionTransformer's "bf16" default while the hub/eval builder
            # (eupe/hub/backbones.py) passes "fp32" — a silent train/eval RoPE-precision mismatch.
            pos_embed_rope_dtype=args.get("pos_embed_rope_dtype", "bf16"),
            qkv_bias=args.qkv_bias,
            layerscale_init=args.layerscale,
            norm_layer=args.norm_layer,
            ffn_layer=args.ffn_layer,
            ffn_bias=args.ffn_bias,
            proj_bias=args.proj_bias,
            n_storage_tokens=args.n_storage_tokens,
            mask_k_bias=args.mask_k_bias,
            untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
            device=device,
        )
        # Honor explicit architecture dims when the cfg provides them, so ViT-T (192/12/3) and a
        # tuned ~1.9B proxy are built as configured instead of silently falling back to the named
        # factory's hardcoded dims. The vit_* factories hardcode embed_dim/depth/num_heads/ffn_ratio
        # and pass **kwargs through, so passing those via kwargs to a factory raises a duplicate-
        # keyword error; when explicit STRUCTURAL dims are present we construct DinoVisionTransformer
        # directly.
        #
        # Detection keys off the STRUCTURAL dims (embed_dim/depth/num_heads) ONLY — never ffn_ratio.
        # ssl_default_config.yaml always supplies ffn_ratio (4.0) while leaving the structural dims to
        # the named factory (e.g. `arch: vit_large` with no embed_dim/depth/num_heads), so keying off
        # ffn_ratio would make `explicit` non-empty and silently build DinoVisionTransformer's
        # 768/12/12 defaults instead of vit_large's 1024/24/16. We still THREAD ffn_ratio into the
        # direct-construction path, so structural-dims configs that also retune the FFN (e.g. the
        # ~1.9B proxy's ffn_ratio=8.0) get it.
        structural = ("embed_dim", "depth", "num_heads")
        explicit = {
            k: args[k]
            for k in structural + ("ffn_ratio",)
            if k in args and args.get(k) is not None
        }
        use_explicit = any(k in explicit for k in structural)

        def _make_vit(extra=None):
            kw = dict(vit_kwargs)
            if extra:
                kw.update(extra)
            if use_explicit:
                kw.update(explicit)
                return vits.DinoVisionTransformer(**kw)
            return vits.__dict__[args.arch](**kw)

        teacher = _make_vit()
        if only_teacher:
            return teacher, teacher.embed_dim
        student = _make_vit({"drop_path_rate": args.drop_path_rate})
        embed_dim = student.embed_dim
    elif "convnext" in args.arch:
        # ConvNeXt students/teachers. get_convnext_arch encodes the canonical depths/dims for the
        # size in the arch name (convnext_tiny/small/base); drop_path_rate / layer-scale / pseudo
        # patch_size come from the cfg. ConvNeXt.forward_features emits the same
        # {x_norm_clstoken (global-avg-pool), x_norm_patchtokens} dict the distillation meta-arch
        # consumes, so no class-token plumbing is needed downstream. (ConvNeXt has no meta-device
        # path, so it materializes on the default device and is sharded/moved by parallelize().)
        from .convnext import get_convnext_arch

        ctor = get_convnext_arch(args.arch)
        cnx_kwargs = dict(
            layer_scale_init_value=args.get("layer_scale_init_value", 1e-6),
            patch_size=args.get("patch_size", None),
        )
        teacher = ctor(drop_path_rate=0.0, **cnx_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = ctor(drop_path_rate=args.get("drop_path_rate", 0.0), **cnx_kwargs)
        embed_dim = student.embed_dim
    else:
        raise NotImplementedError(f"Unrecognized architecture {args.arch}")
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher: bool = False):
    outputs = build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=(
            cfg.crops.global_crops_size
            if isinstance(cfg.crops.global_crops_size, int)
            else max(cfg.crops.global_crops_size)
        ),
        device="meta",
    )
    if only_teacher:
        teacher, embed_dim = outputs
        return teacher, embed_dim
    else:
        student, teacher, embed_dim = outputs
        return student, teacher, embed_dim


def extract_backbone_state_dict(checkpoint) -> dict:
    """Return a plain (un-prefixed) backbone state dict from any EUPE checkpoint layout.

    Handles all three shapes the codebase produces/consumes:
      * eupe.train.train._save_checkpoint payload: {"teacher": {"teacher.<k>": v}, "optimizer": ...}
      * a flat 'teacher.'-prefixed state dict: {"teacher.<k>": v}
      * a plain backbone state dict: {"<k>": v}
    Previously callers filtered ``state_dict.items()`` for keys starting with 'teacher.', which on the
    nested payload above matched nothing (the top-level keys are 'teacher'/'optimizer'/'iteration')
    and produced an empty dict — breaking proxy and eval reloads.
    """
    sd = checkpoint
    # Unwrap the {"teacher": {...}, "optimizer": ..., "iteration": ...} training payload.
    if isinstance(sd, dict) and isinstance(sd.get("teacher"), dict):
        sd = sd["teacher"]
    elif isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    # Strip a flat 'teacher.' prefix if present.
    if any(isinstance(k, str) and k.startswith("teacher.") for k in sd):
        sd = {k[len("teacher.") :]: v for k, v in sd.items() if k.startswith("teacher.")}
    # Strip FSDP / activation-checkpoint / torch.compile name decorations the student may have
    # carried at save time (it is saved while FSDP2/AC/compile-wrapped). Mirrors
    # eupe.train.param_groups._remove_fsdp_compile_names so keys match a freshly-built backbone.
    sd = {_strip_wrapper_names(k): v for k, v in sd.items()}
    return sd


def _strip_wrapper_names(name: str) -> str:
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("_checkpoint_wrapped_module.", "")
    name = name.replace("parametrizations.", "")
    name = name.removesuffix(".original")
    name = name.replace("_orig_mod.", "")
    return name


def build_model_for_eval(
    config,
    pretrained_weights: Union[str, Path] | None,
):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights is None or pretrained_weights == "":
        logger.info("No pretrained weights")
        model.init_weights()
    else:
        logger.info("PyTorch consolidated checkpoint")
        model.to_empty(device="cuda")
        state_dict = extract_backbone_state_dict(torch.load(pretrained_weights, map_location="cpu"))
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
