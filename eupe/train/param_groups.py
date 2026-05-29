# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""AdamW parameter groups for distillation (mirrors dinov3/train/param_groups.py).

Applies layer-wise LR decay across ViT blocks (optim.layerwise_decay), a patch-embed LR
multiplier (optim.patch_embed_lr_mult=0.2), and no weight decay on norms/biases/tokens.
"""
import logging
from collections import defaultdict
from typing import Dict, List

from torch import nn

logger = logging.getLogger("eupe")


def _remove_fsdp_compile_names(name: str) -> str:
    # Ported from refs/dinov3/dinov3/train/param_groups.py:remove_fsdp_compile_names — verbatim.
    name = name.replace("_fsdp_wrapped_module.", "")  # Added by FSDP
    name = name.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing for xFSDP
    name = name.replace("parametrizations.", "")  # Added by xFSDP
    name = name.removesuffix(".original")  # Added by xFSDP
    name = name.replace("module.", "")  # Added by xFSDP
    name = name.replace("_orig_mod.", "")  # Added by torch.compile
    return name


def _layer_index(name: str, num_blocks: int) -> int:
    # Ported from refs/dinov3/dinov3/train/param_groups.py:get_vit_lr_decay_rate — adapted to EUPE
    # vision_transformer naming (patch_embed / blocks.<i> / cls_token / storage_tokens /
    # mask_token); EUPE has no "backbone" prefix and the wrapped model exposes blocks directly.
    # Returns layer id in [0, num_blocks + 1]: patch_embed & learned tokens/embeddings = 0,
    # block i = i + 1, everything else (norm/head) = last layer = num_blocks + 1.
    last = num_blocks + 1
    if (
        "patch_embed" in name
        or "pos_embed" in name
        or "rope_embed" in name
        or "cls_token" in name
        or "storage_tokens" in name
        or "mask_token" in name
    ):
        return 0
    if "blocks." in name:
        # name like "...blocks.<i>...." -> grab the integer right after "blocks."
        return int(name[name.find("blocks.") :].split(".")[1]) + 1
    return last


def _num_blocks(model: nn.Module) -> int:
    # EUPE DinoVisionTransformer exposes a `blocks` ModuleList (n_blocks == depth). Mirror dinov3's
    # branch order: prefer a wrapped `.module`, then direct `.blocks`, then `.backbone.blocks`.
    target = getattr(model, "module", model)
    if hasattr(target, "blocks"):
        return len(target.blocks)
    if hasattr(target, "backbone") and hasattr(target.backbone, "blocks"):
        return len(target.backbone.blocks)
    return 0


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
    # Ported from refs/dinov3/dinov3/train/param_groups.py:get_params_groups_with_decay —
    # divergence: EUPE resolves concrete `lr`/`weight_decay` into each group (the frozen interface
    # passes lr/wd directly) instead of deferring multiplier application; layer ids adapted to
    # EUPE naming; weight decay zeroed for any ndim==1 param (norms/biases/layerscale gamma) and
    # any param whose name contains "token".
    num_blocks = _num_blocks(model)
    last = num_blocks + 1
    force_is_backbone = num_blocks > 0

    groups: List[Dict] = []
    for raw_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name = _remove_fsdp_compile_names(raw_name)

        idx = _layer_index(name, num_blocks) if force_is_backbone else last
        lr_mult = layerwise_decay ** (last - idx)
        if "patch_embed" in name:
            lr_mult *= patch_embed_lr_mult

        wd_mult = 1.0
        # No weight decay on 1-D params (norms/biases/layerscale gamma) or learned tokens.
        if param.ndim <= 1 or "token" in name:
            wd_mult = 0.0

        d = {
            "name": name,
            "params": param,
            "lr_mult": lr_mult,
            "wd": wd * wd_mult,
            "lr": lr * lr_mult,
            "weight_decay": wd * wd_mult,
        }
        groups.append(d)
        logger.info(f"{name}: lr_mult={lr_mult}, wd={d['wd']}")

    return groups


def fuse_params_groups(groups: List[Dict]) -> List[Dict]:
    """Merge param groups that share (lr_mult, wd) to reduce optimizer overhead."""
    # Ported from refs/dinov3/dinov3/train/param_groups.py:fuse_params_groups — divergence: fuse
    # key is (lr_mult, wd) per the EUPE frozen interface, each param dict carries a single Tensor
    # under "params" so we collect them into a list; resolved lr/weight_decay carried through.
    fused: Dict[str, Dict] = defaultdict(lambda: {"params": []})
    keys = ("lr_mult", "wd")
    for d in groups:
        identifier = "_".join(f"{k}{d[k]}" for k in keys)
        bucket = fused[identifier]
        for k in keys:
            bucket[k] = d[k]
        bucket["lr"] = d["lr"]
        bucket["weight_decay"] = d["weight_decay"]
        bucket["params"].append(d["params"])
    return list(fused.values())
