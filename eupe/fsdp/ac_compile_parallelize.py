# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""FSDP2 sharding + activation checkpointing + torch.compile (mirrors dinov3/fsdp).

Reads cfg.compute_precision (param_dtype=bf16, reduce_dtype=fp32, sharding_strategy=SHARD_GRAD_OP)
and cfg.train.{checkpointing, checkpointing_full, compile}.

Aligned to dinov3's FSDP2 path (``torch.distributed._composable.fsdp.fully_shard``): each transformer
block (or ConvNeXt block-within-stage / downsample unit) is sharded independently with forward /
backward prefetch, the whole backbone is sharded last, and meta-device init is materialized via
``to_empty`` so the caller's ``init_weights()`` can fill each local shard. The legacy
``cfg.compute_precision.sharding_strategy`` enum name is mapped onto fully_shard's
``reshard_after_forward`` (FULL_SHARD -> True / SHARD_GRAD_OP -> False). FSDP2's MixedPrecisionPolicy
casts params/grads only — buffers (e.g. the fp32 RoPE periods) are left untouched, matching dinov3.
"""
import logging
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from eupe.layers.block import SelfAttentionBlock  # noqa: F401  (documents the ViT block unit)

logger = logging.getLogger("eupe")

# Map config dtype strings to torch dtypes (mirrors dinov3's DTYPE_MAP).
_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

# FULL_SHARD (ZeRO-3) reshards params after forward; SHARD_GRAD_OP (ZeRO-2) keeps them gathered.
_RESHARD_AFTER_FORWARD = {"FULL_SHARD": True, "SHARD_GRAD_OP": False, "NO_SHARD": False}


def _is_convnext(model: nn.Module) -> bool:
    return isinstance(getattr(model, "stages", None), nn.ModuleList) and isinstance(
        getattr(model, "downsample_layers", None), nn.ModuleList
    )


def _get_activation_checkpoint_wrapper(full: bool):
    # Ported from refs/dinov3/dinov3/fsdp/ac_compile_parallelize.py:get_activation_checkpoint_wrapper
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    if full:
        logger.info("activation checkpointing: full policy")
        return checkpoint_wrapper
    _save_list = [
        # mm
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_mm.default,
        # attentions
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
    ]
    logger.info("activation checkpointing: selective policy")
    return partial(
        checkpoint_wrapper,
        context_fn=partial(create_selective_checkpoint_contexts, _save_list),
        preserve_rng_state=True,
    )


def apply_activation_checkpointing(model: nn.Module, full: bool = False) -> nn.Module:
    """Wrap each transformer block (ViT) or each block-within-stage + downsample unit (ConvNeXt)."""
    # Ported from refs/dinov3/.../ac_compile_parallelize.py:activation_checkpoint_{transformer,convnext}
    # — per-block granularity (ConvNeXt wraps inside each stage, matching dinov3, not whole stages).
    wrapper = _get_activation_checkpoint_wrapper(full)
    if _is_convnext(model):
        for stage_id, stage in enumerate(model.stages):
            for block_id, block in enumerate(stage):
                model.stages[stage_id][block_id] = wrapper(block)
        for dsl_id, dsl in enumerate(model.downsample_layers):
            model.downsample_layers[dsl_id] = wrapper(dsl)
    else:
        for block_id, block in enumerate(model.blocks):
            model.blocks[block_id] = wrapper(block)
    return model


def _compile_unit(module: nn.Module) -> nn.Module:
    module.compile()
    return module


def apply_compile(model: nn.Module) -> nn.Module:
    """torch.compile each transformer block (ViT) or each stage + downsample unit (ConvNeXt)."""
    # Ported from refs/dinov3/.../ac_compile_parallelize.py:compile_{transformer,convnext} — compile
    # the per-block (ViT) / per-stage (ConvNeXt) forward in place so dynamo reuses the graph.
    if _is_convnext(model):
        for stage_id, stage in enumerate(model.stages):
            model.stages[stage_id] = _compile_unit(stage)
        for dsl_id, dsl in enumerate(model.downsample_layers):
            model.downsample_layers[dsl_id] = _compile_unit(dsl)
    else:
        for block_id, block in enumerate(model.blocks):
            model.blocks[block_id] = _compile_unit(block)
    return model


def _fully_shard_transformer(model: nn.Module, fsdp_config: dict, reshard_after_forward) -> None:
    # Ported from refs/dinov3/.../ac_compile_parallelize.py:fsdp_transformer — shard every block, set
    # adjacent-block forward/backward prefetch, then shard the whole backbone last.
    from torch.distributed._composable.fsdp import fully_shard

    blocks = model.blocks
    for block_id, block in enumerate(blocks):
        blocks[block_id] = fully_shard(block, reshard_after_forward=reshard_after_forward, **fsdp_config)
    for prev_block, next_block in zip(blocks, blocks[1:]):
        prev_block.set_modules_to_forward_prefetch([next_block])
        next_block.set_modules_to_backward_prefetch([prev_block])
    fully_shard(model, reshard_after_forward=True, **fsdp_config)


def _fully_shard_convnext(model: nn.Module, fsdp_config: dict, reshard_after_forward) -> None:
    # Ported from refs/dinov3/.../ac_compile_parallelize.py:fsdp_convnext — shard each stage + each
    # downsample layer, cross-prefetch downsample<->stage, then shard the whole backbone last.
    from torch.distributed._composable.fsdp import fully_shard

    stages = model.stages
    for stage_id, stage in enumerate(stages):
        stages[stage_id] = fully_shard(stage, reshard_after_forward=reshard_after_forward, **fsdp_config)
    downsample_layers = model.downsample_layers
    for dsl_id, dsl in enumerate(downsample_layers):
        downsample_layers[dsl_id] = fully_shard(
            dsl, reshard_after_forward=reshard_after_forward, **fsdp_config
        )
    for dsl, stage in zip(downsample_layers, stages):
        dsl.set_modules_to_forward_prefetch([stage])
        stage.set_modules_to_backward_prefetch([dsl])
    fully_shard(model, reshard_after_forward=True, **fsdp_config)


def parallelize(model: nn.Module, cfg) -> nn.Module:
    """FSDP2-shard the backbone per cfg.compute_precision, optionally AC + compile; return it.

    Mirrors dinov3 ``ac_compile_parallelize``: 1/ activation checkpointing, 2/ compile, 3/ per-block
    ``fully_shard`` (+ prefetch) then whole-backbone ``fully_shard``, 4/ ``to_empty`` to materialize
    the sharded meta params on cuda. The caller (train.py) then runs ``init_weights()`` to fill each
    local shard. The non-default training entry point (``forward_features``) is registered with FSDP2
    so its params are all-gathered/resharded around the forward.
    """
    # — divergence from the prior FSDP1 port: this now uses FSDP2 fully_shard (DTensor sharding),
    #   which is the meta-init path dinov3 uses; cfg.compute_precision.sharding_strategy is mapped to
    #   reshard_after_forward rather than a ShardingStrategy enum.
    import eupe.distributed as eupe_distributed
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.fsdp import register_fsdp_forward_method

    logger.info("DISTRIBUTED FSDP2 -- preparing model for distributed training")

    # Order matches dinov3: 1/ activation checkpointing, 2/ compile, 3/ FSDP.
    if getattr(cfg.train, "checkpointing", False):
        apply_activation_checkpointing(model, full=getattr(cfg.train, "checkpointing_full", False))
    if getattr(cfg.train, "compile", False):
        apply_compile(model)

    # MixedPrecisionPolicy casts params/grads only; buffers (fp32 RoPE periods) are left as-is.
    mp_policy = MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP[cfg.compute_precision.param_dtype],
        reduce_dtype=_DTYPE_MAP[cfg.compute_precision.reduce_dtype],
    )
    reshard_after_forward = _RESHARD_AFTER_FORWARD.get(cfg.compute_precision.sharding_strategy, True)
    on_meta = any(p.is_meta for p in model.parameters())

    if not (dist.is_available() and dist.is_initialized()):
        # No process group (e.g. single-process smoke test): can't shard; materialize so the rest runs.
        logger.warning("parallelize: no initialized process group -- skipping FSDP2 shard")
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        if on_meta:
            model.to_empty(device=device)
        elif torch.cuda.is_available():
            model = model.cuda()
        return model

    # In multidistillation each rank builds ONLY its assigned student, so the FSDP mesh must span
    # that student's process SUBGROUP, not the whole world — sharding a student across ranks that
    # hold a different student would deadlock / mis-shard the collectives. get_process_subgroup()
    # returns this rank's subgroup in multidist runs and the default (world) group in single-student
    # runs (Stage 1), so DeviceMesh.from_group is correct in both cases.
    process_group = eupe_distributed.get_process_subgroup()
    if eupe_distributed.get_subgroup_size() < dist.get_world_size():
        mesh = DeviceMesh.from_group(process_group, "cuda")
    else:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}
    if _is_convnext(model):
        _fully_shard_convnext(model, fsdp_config, reshard_after_forward)
    else:
        _fully_shard_transformer(model, fsdp_config, reshard_after_forward)

    # FSDP2 only hooks the module's `forward`; training calls student.forward_features(...), and eval
    # uses get_intermediate_layers, so register both as managed forward methods (all-gather/reshard).
    for method_name in ("forward_features", "get_intermediate_layers"):
        if hasattr(model, method_name):
            register_fsdp_forward_method(model, method_name)

    # Meta-device init (dinov3 step 4): materialize the now-sharded params on cuda as empty storage;
    # train.py then calls model.student.init_weights() to initialize each local shard.
    if on_meta:
        model.to_empty(device="cuda")
    return model
