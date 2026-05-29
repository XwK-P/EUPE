# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

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
