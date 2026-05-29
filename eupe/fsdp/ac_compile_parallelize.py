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
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from eupe.layers.block import SelfAttentionBlock

logger = logging.getLogger("eupe")

# Map config dtype strings to torch dtypes (mirrors dinov3's DTYPE_MAP).
_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _iter_transformer_blocks(model: nn.Module):
    """Yield (parent_list, index) for every transformer block in a ViT-style backbone."""
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, nn.ModuleList):
        for block_id in range(len(blocks)):
            yield blocks, block_id


def _iter_convnext_units(model: nn.Module):
    """Yield (parent_list, index) for ConvNeXt stages + downsample layers."""
    stages = getattr(model, "stages", None)
    downsample_layers = getattr(model, "downsample_layers", None)
    if isinstance(stages, nn.ModuleList):
        for stage_id in range(len(stages)):
            yield stages, stage_id
    if isinstance(downsample_layers, nn.ModuleList):
        for dsl_id in range(len(downsample_layers)):
            yield downsample_layers, dsl_id


def _is_convnext(model: nn.Module) -> bool:
    return isinstance(getattr(model, "stages", None), nn.ModuleList) and isinstance(
        getattr(model, "downsample_layers", None), nn.ModuleList
    )


def apply_activation_checkpointing(model: nn.Module, full: bool = False) -> nn.Module:
    """Wrap transformer blocks (or everything, if full) with activation checkpointing."""
    # Ported from refs/dinov3/dinov3/fsdp/ac_compile_parallelize.py:
    # activation_checkpoint_transformer / activation_checkpoint_convnext / get_activation_checkpoint_wrapper
    # — divergence: collapsed into one fn taking a bare backbone with a `full` flag (no cfg object,
    #   FSDP1 instead of FSDP2), and selective-vs-full policy chosen by `full`.
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    if full:
        _checkpointing_wrapper = checkpoint_wrapper
        logger.info("activation checkpointing: full policy")
    else:
        _save_list = [
            # mm
            torch.ops.aten.mm.default,
            torch.ops.aten._scaled_mm.default,
            # attentions
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
        ]
        _checkpointing_wrapper = partial(
            checkpoint_wrapper,
            context_fn=partial(create_selective_checkpoint_contexts, _save_list),
            preserve_rng_state=True,
        )
        logger.info("activation checkpointing: selective policy")

    if _is_convnext(model):
        for parent, idx in _iter_convnext_units(model):
            parent[idx] = _checkpointing_wrapper(parent[idx])
    else:
        for parent, idx in _iter_transformer_blocks(model):
            parent[idx] = _checkpointing_wrapper(parent[idx])
    return model


def apply_compile(model: nn.Module) -> nn.Module:
    """torch.compile the per-block forward. See dinov3 ac_compile_parallelize."""
    # Ported from refs/dinov3/dinov3/fsdp/ac_compile_parallelize.py:
    # compile_transformer / compile_convnext / wrap_compile_block
    # — divergence: bare backbone (no cfg/cudagraphs); compile each block/stage in-place with the
    #   default backend so dynamo can reuse the per-block graph across the ModuleList.
    if _is_convnext(model):
        for parent, idx in _iter_convnext_units(model):
            parent[idx].compile()
    else:
        for parent, idx in _iter_transformer_blocks(model):
            parent[idx].compile()
    return model


def parallelize(model: nn.Module, cfg) -> nn.Module:
    """FSDP-wrap per cfg.compute_precision, optionally activation-ckpt + compile; return wrapped model.

    Build a MixedPrecision policy from param_dtype/reduce_dtype; choose ShardingStrategy from
    cfg.compute_precision.sharding_strategy; init on meta device then to_empty/cuda.
    """
    # Ported from refs/dinov3/dinov3/fsdp/ac_compile_parallelize.py:ac_compile_parallelize
    # — divergence: dinov3 uses FSDP2 (fully_shard) on a backbone ModuleDict; here we use the FSDP1
    #   FullyShardedDataParallel wrapper because EUPE's cfg.compute_precision.sharding_strategy names
    #   a ShardingStrategy enum (SHARD_GRAD_OP/FULL_SHARD) and the entry point hands us a bare backbone.
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")

    # Order matches dinov3: 1/ activation checkpointing, 2/ compile, 3/ FSDP.
    if getattr(cfg.train, "checkpointing", False):
        apply_activation_checkpointing(model, full=getattr(cfg.train, "checkpointing_full", False))
    if getattr(cfg.train, "compile", False):
        apply_compile(model)

    # 1/ MixedPrecision policy from cfg.compute_precision.{param_dtype, reduce_dtype}.
    mp_policy = MixedPrecision(
        param_dtype=_DTYPE_MAP[cfg.compute_precision.param_dtype],
        reduce_dtype=_DTYPE_MAP[cfg.compute_precision.reduce_dtype],
        buffer_dtype=_DTYPE_MAP[cfg.compute_precision.param_dtype],
    )

    # 2/ ShardingStrategy enum lookup by name (SHARD_GRAD_OP / FULL_SHARD / NO_SHARD / ...).
    sharding_strategy = ShardingStrategy[cfg.compute_precision.sharding_strategy]

    # 3/ Auto-wrap each transformer block (or ConvNeXt stage/downsample unit) as its own FSDP
    #    instance, so each is sharded independently like dinov3's per-block fully_shard. Activation
    #    checkpointing (if applied above) wraps the blocks in CheckpointWrapper, so include it in the
    #    wrap policy too.
    if _is_convnext(model):
        # RECONSTRUCTED (unverified): ConvNeXt stage/downsample units are nn.Sequential, so wrap by
        # the leaf module classes that compose them; fall back to wrapping every nn.Sequential unit.
        wrap_classes = {nn.Sequential, CheckpointWrapper}
    else:
        wrap_classes = {SelfAttentionBlock, CheckpointWrapper}
    auto_wrap_policy = ModuleWrapPolicy(wrap_classes)

    # 4/ Meta-device init support: if the model was built on the meta device, FSDP needs a
    #    param_init_fn to materialize each shard on cuda via to_empty before init_weights runs.
    on_meta = any(p.is_meta for p in model.parameters())

    def _param_init_fn(module: nn.Module) -> None:
        module.to_empty(device=torch.cuda.current_device(), recurse=False)

    fsdp_kwargs = dict(
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
    )
    if on_meta:
        fsdp_kwargs["param_init_fn"] = _param_init_fn

    if dist.is_available() and dist.is_initialized():
        model = FSDP(model, **fsdp_kwargs)
    else:
        # RECONSTRUCTED (unverified): FSDP requires an initialized process group; if none is present
        # (e.g. single-process smoke test), skip sharding and just move the model onto cuda so the
        # rest of the pipeline still runs.
        logger.warning("parallelize: no initialized process group -- skipping FSDP wrap")
        if on_meta:
            model.to_empty(device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        elif torch.cuda.is_available():
            model = model.cuda()

    return model
