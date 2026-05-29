# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

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
