# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Distillation training entry point.

Launched via:  python -m eupe.run.submit eupe/train/train.py --nodes N [--multi-distillation] \
                 --config-file eupe/configs/train/stageX_*.yaml --output-dir <OUT>
Builds the meta-arch on meta device, FSDP-wraps it, runs the 500-iter normalizer warmup, then
iterates the LVD+ImageNet sampler, applying cosine LR/WD/momentum schedules and periodic
checkpointing. Routes to MultiDistillationMetaArch when cfg.multidistillation.enabled.
"""
import argparse
import logging
import math
import os
from pathlib import Path

import torch

import eupe.distributed as distributed
from eupe.configs import EupeSetupArgs, setup_config, setup_multidistillation
from eupe.data.distillation_loaders import make_distillation_data_loader
from eupe.fsdp import parallelize
from eupe.run.init import job_context
from eupe.train.cosine_lr_scheduler import CosineScheduler
from eupe.train.param_groups import fuse_params_groups, get_params_groups_with_decay

logger = logging.getLogger("eupe")


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """--config-file, --output-dir, --multi-distillation, opts (mirrors dinov3/train/train.py)."""
    # Ported from refs/dinov3/dinov3/train/train.py:get_args_parser — divergence: dropped the
    # eval-only / ibot-test / profiling / gram flags that EUPE distillation does not use; kept
    # --config-file, --output-dir, --no-resume, --seed, --multi-distillation and trailing opts.
    parser = argparse.ArgumentParser("EUPE distillation training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="./local_eupe",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--multi-distillation",
        action="store_true",
        help="run multi-distillation (1 proxy -> M students on rank-subgroups)",
    )
    parser.add_argument(
        "opts",
        help=(
            "Modify config options at the end of the command using dotted "
            '"path.key=value" pairs (OmegaConf from_cli).'
        ).strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def build_optimizer(cfg, params_groups):
    """torch.optim.AdamW(betas=(adamw_beta1, adamw_beta2)). See dinov3."""
    # Ported from refs/dinov3/dinov3/train/train.py:build_optimizer — verbatim: AdamW over the
    # supplied (already lr/wd-resolved) param groups, betas read from cfg.optim.adamw_beta1/2.
    return torch.optim.AdamW(
        params_groups,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )


def _total_iters(cfg) -> int:
    """Total training iterations = OFFICIAL_EPOCH_LENGTH * optim.epochs."""
    return int(cfg.train.OFFICIAL_EPOCH_LENGTH) * int(cfg.optim.epochs)


def _warmup_iters(cfg, schedule_cfg=None) -> int:
    """Resolve warmup iterations from per-schedule warmup_epochs (schedules-v2) or optim.warmup_epochs."""
    epoch_len = int(cfg.train.OFFICIAL_EPOCH_LENGTH)
    if schedule_cfg is not None and "warmup_epochs" in schedule_cfg:
        return int(schedule_cfg.warmup_epochs) * epoch_len
    if "warmup_epochs" in cfg.optim:
        return int(cfg.optim.warmup_epochs) * epoch_len
    return 0


def build_schedulers(cfg) -> dict:
    """Return {"lr","wd","momentum","teacher_temp"} CosineSchedulers from cfg (schedules-v2 aware).

    schedules-v2: when a `schedules:` block is present, read the peak value directly
    (cfg.schedules.lr.peak etc.) so the implicit LR scaling in apply_scaling_rules_to_cfg is
    bypassed and reproducers hit the paper's exact peak LR regardless of GPU count.
    """
    # Ported from refs/dinov3/dinov3/train/train.py:build_schedulers / build_schedulers_v2 —
    # divergence: EUPE's cosine_lr_scheduler only exposes CosineScheduler (no
    # linear_warmup_cosine_decay helper), so schedules-v2 is realized by reading
    # cfg.schedules.<name>.{start,peak,end,warmup_epochs} straight into CosineScheduler kwargs
    # (peak == base_value). The DINO/iBOT teacher_temp + last_layer terms are gone (distillation
    # has no EMA teacher), but the frozen interface keeps "momentum"/"teacher_temp" keys for
    # callers; they default to flat schedules at the configured value.
    total_iters = _total_iters(cfg)
    logger.info("Total training iterations %d", total_iters)
    schedules_v2 = "schedules" in cfg

    if schedules_v2:
        logger.info("Using schedules v2 (peak-aware)")
        sched = cfg.schedules

        lr_cfg = sched.lr
        lr = CosineScheduler(
            base_value=lr_cfg.peak,  # schedules-v2: read the peak LR directly (no scaling rule)
            final_value=lr_cfg.get("end", 0.0),
            total_iters=total_iters,
            warmup_iters=_warmup_iters(cfg, lr_cfg),
            start_warmup_value=lr_cfg.get("start", 0.0),
        )

        if "weight_decay" in sched:
            wd_cfg = sched.weight_decay
            wd = CosineScheduler(
                base_value=wd_cfg.peak,
                final_value=wd_cfg.get("end", wd_cfg.peak),
                total_iters=total_iters,
                warmup_iters=_warmup_iters(cfg, wd_cfg),
                start_warmup_value=wd_cfg.get("start", wd_cfg.peak),
            )
        else:
            # No dedicated WD schedule: hold the AdamW weight decay flat.
            wd_const = float(cfg.optim.get("weight_decay", 0.0))
            wd = CosineScheduler(base_value=wd_const, final_value=wd_const, total_iters=total_iters)

        if "momentum" in sched:
            mom_cfg = sched.momentum
            momentum = CosineScheduler(
                base_value=mom_cfg.peak,
                final_value=mom_cfg.get("end", mom_cfg.peak),
                total_iters=total_iters,
                warmup_iters=_warmup_iters(cfg, mom_cfg),
                start_warmup_value=mom_cfg.get("start", mom_cfg.peak),
            )
        else:
            momentum = CosineScheduler(base_value=0.0, final_value=0.0, total_iters=total_iters)

        if "teacher_temp" in sched:
            tt_cfg = sched.teacher_temp
            teacher_temp = CosineScheduler(
                base_value=tt_cfg.peak,
                final_value=tt_cfg.get("end", tt_cfg.peak),
                total_iters=total_iters,
                warmup_iters=_warmup_iters(cfg, tt_cfg),
                start_warmup_value=tt_cfg.get("start", tt_cfg.peak),
            )
        else:
            teacher_temp = CosineScheduler(base_value=0.0, final_value=0.0, total_iters=total_iters)

    else:
        # schedules-v1: endpoints come from cfg.optim (lr/min_lr/weight_decay[_end]). The scaling
        # rule has already been folded into cfg.optim.lr by apply_scaling_rules_to_cfg.
        lr = CosineScheduler(
            base_value=cfg.optim.lr,
            final_value=cfg.optim.get("min_lr", 0.0),
            total_iters=total_iters,
            warmup_iters=_warmup_iters(cfg),
            start_warmup_value=0.0,
        )
        wd = CosineScheduler(
            base_value=cfg.optim.get("weight_decay", 0.0),
            final_value=cfg.optim.get("weight_decay_end", cfg.optim.get("weight_decay", 0.0)),
            total_iters=total_iters,
        )
        # Distillation has no EMA teacher; momentum/teacher_temp are flat placeholders kept so the
        # frozen-interface dict shape is stable for callers that index them.
        momentum = CosineScheduler(base_value=0.0, final_value=0.0, total_iters=total_iters)
        teacher_temp = CosineScheduler(base_value=0.0, final_value=0.0, total_iters=total_iters)

    logger.info("Schedulers ready.")
    return {"lr": lr, "wd": wd, "momentum": momentum, "teacher_temp": teacher_temp}


def apply_optim_scheduler(optimizer, schedulers, iteration: int) -> None:
    """Set per-group lr/wd from schedulers[iteration]."""
    # Ported from refs/dinov3/dinov3/train/train.py:apply_optim_scheduler — divergence: schedulers
    # is the {"lr","wd",...} dict returned by build_schedulers (not a positional tuple), and EUPE's
    # param groups carry "lr_mult"/"wd" (resolved by get_params_groups_with_decay) rather than
    # dinov3's lr_multiplier/wd_multiplier + is_last_layer. We re-scale the base lr/wd by each
    # group's stored multiplier so the layer-wise decay is preserved across the schedule.
    lr = schedulers["lr"][iteration]
    wd = schedulers["wd"][iteration]
    for param_group in optimizer.param_groups:
        lr_mult = param_group.get("lr_mult", 1.0)
        # wd_mult: groups that were assigned wd==0 (norms/biases/tokens) must stay at 0; scale the
        # scheduled wd by the ratio the group was originally resolved to (0.0 or 1.0).
        wd_mult = 1.0 if param_group.get("wd", 0.0) != 0.0 else 0.0
        param_group["lr"] = lr * lr_mult
        param_group["weight_decay"] = wd * wd_mult


def build_data_loader(cfg, start_iteration: int = 0):
    """eupe.data.distillation_loaders.make_distillation_data_loader(cfg).

    start_iteration > 0 (resume) fast-forwards the mixed sampler so the data stream continues from
    where the preempted run left off instead of replaying from batch 0.
    """
    return make_distillation_data_loader(cfg, start_iteration=start_iteration)


def _state_dict_with_teacher_prefix(student) -> dict:
    """Return the student state dict re-keyed under a 'teacher.' prefix.

    Mirrors eupe/models/__init__.build_model_for_eval, which loads checkpoints by stripping a
    'teacher.' prefix; saving the trained student that way lets ProxyTeacher / eval reload it.
    """
    student_sd = student.state_dict()
    teacher_sd = {}
    for k, v in student_sd.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.full_tensor()
        teacher_sd[f"teacher.{k}"] = v
    return teacher_sd


def _consolidated_optimizer_state(model, optimizer):
    """Return a full (rank-0-complete) optimizer state dict that can be reloaded on resume.

    Under FSDP2 (fully_shard) the student's AdamW moments are DTensors sharded over the student's
    process subgroup; a bare ``optimizer.state_dict()`` would persist only this rank's shard wrapped in
    a stale-mesh DTensor. ``get_optimizer_state_dict(..., full_state_dict=True)`` consolidates them (and
    the non-sharded adapter moments) into a full CPU state dict. It is a COLLECTIVE over the student's
    mesh and must run on every subgroup rank. Best-effort: on any failure we return None and the
    checkpoint omits optimizer state (resume then re-initializes AdamW moments rather than crashing).
    """
    if not distributed.is_enabled():
        return optimizer.state_dict()
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

        return get_optimizer_state_dict(
            model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    except Exception as e:  # never let checkpointing crash a training run
        logger.warning("optimizer-state consolidation failed (%s); checkpoint will omit optimizer state", e)
        return None


def _restore_optimizer_state(model, optimizer, state) -> bool:
    """Load a consolidated optimizer state dict back onto the (re-sharded) optimizer. Returns success."""
    if state is None:
        return False
    if not distributed.is_enabled():
        optimizer.load_state_dict(state)
        return True
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

    set_optimizer_state_dict(
        model, optimizer, optim_state_dict=state, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    return True


def _find_latest_checkpoint(output_dir):
    """Return (path, iteration) of the highest-numbered training_<it>.pth under <output_dir>/ckpt.

    Used to resume a preempted run (paper Stage 2/3 are 390k/100k iters — effectively guaranteed to be
    preempted on a real cluster). Returns (None, -1) when no checkpoint directory / file exists.
    """
    ckpt_dir = Path(output_dir, "ckpt").expanduser()
    if not ckpt_dir.is_dir():
        return None, -1
    best_path, best_it = None, -1
    for p in ckpt_dir.glob("training_*.pth"):
        try:
            it = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        if it > best_it:
            best_path, best_it = p, it
    return best_path, best_it


def _save_checkpoint(cfg, model, optimizer, iteration: int) -> None:
    """Save the trained student (+optimizer / frozen normalizer stats / iteration) for eval, proxy
    reuse, and resume. The student weights are re-keyed under 'teacher.' so build_model_for_eval /
    ProxyTeacher can reload them directly.

    CRITICAL ordering (was a multi-rank deadlock): consolidating the weights (DTensor.full_tensor) and
    the optimizer state (get_optimizer_state_dict) are COLLECTIVES every subgroup rank must enter. We
    therefore consolidate on ALL ranks FIRST, then gate only the torch.save on the subgroup-main
    process, then barrier. The previous code returned early on non-main ranks BEFORE the full_tensor()
    all-gather, so only rank-0 entered the collective and the run hung (NCCL timeout).
    """
    # Ported from refs/dinov3/dinov3/train/train.py:do_test teacher save — same "consolidate on all
    # ranks, write on main, barrier" ordering.
    teacher_sd = _state_dict_with_teacher_prefix(model.student)
    optimizer_sd = _consolidated_optimizer_state(model, optimizer)
    normalizer_sd = model.normalizers.state_dict()  # frozen per-teacher mean/std (small, plain buffers)

    if not distributed.is_enabled() or distributed.is_subgroup_main_process():
        ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"training_{iteration}.pth"
        payload = {
            "teacher": teacher_sd,
            "optimizer": optimizer_sd,
            "normalizers": normalizer_sd,
            "iteration": iteration,
        }
        torch.save(payload, ckpt_path)
        logger.info("Saved checkpoint: %s", ckpt_path)
    # Barrier so non-main ranks do not race ahead — in particular so they do not tear down the process
    # group (job_context teardown -> destroy_process_group) before rank-0 finishes the final write.
    if distributed.is_enabled():
        torch.distributed.barrier()


def _load_student_init_checkpoint(student, checkpoint_path) -> None:
    """Initialize the student from a prior-stage checkpoint (Stage 3 finetunes from Stage 2; paper §3.1).

    Called BEFORE FSDP sharding (so fully_shard shards the loaded weights) and AFTER init_weights()
    (so non-persistent buffers such as the RoPE periods are populated, then params are overwritten).
    Reuses eupe.models.extract_backbone_state_dict to accept the {"teacher": {...}} training payload
    and strip FSDP/AC/compile name decorations.
    """
    from eupe.models import extract_backbone_state_dict

    logger.info("Initializing student from prior-stage checkpoint: %s", checkpoint_path)
    state_dict = extract_backbone_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            "student init from %s: %d missing / %d unexpected keys (missing[:3]=%s unexpected[:3]=%s)",
            checkpoint_path, len(missing), len(unexpected), list(missing)[:3], list(unexpected)[:3],
        )
    else:
        logger.info("student init from checkpoint: all keys matched")


def _clip_gradients(model, max_norm: float) -> None:
    """Clip the student grads and the (separately-held) adapter grads, each to ``max_norm``.

    Under FSDP2 (``fully_shard``) the student params are DTensors; ``torch.nn.utils.clip_grad_norm_``
    is DTensor-aware and reduces the correct *global* norm across the student's shards. The adapter
    heads live outside the FSDP unit and are clipped in a SEPARATE call. This is per-unit clipping
    (each unit bounded to ``max_norm`` independently) — it matches dinov3, which also clips each
    student unit separately, and is NOT a single joint global norm over student+adapters.
    """
    torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm)
    adapter_grads = [p for p in model.adapters.parameters() if p.grad is not None]
    if adapter_grads:
        torch.nn.utils.clip_grad_norm_(adapter_grads, max_norm)


def do_train(cfg, model, *, resume_optimizer_state=None, start_iteration: int = 0,
             skip_normalizer_init: bool = False) -> None:
    """Main loop: init_normalizer -> for it in range(max_iter): forward_backward; schedule; checkpoint.

    Resume: when resuming a preempted run, ``start_iteration`` continues the cosine LR/WD schedule and
    the data stream where it left off, ``resume_optimizer_state`` restores the AdamW moments (best
    effort), and ``skip_normalizer_init`` reuses the frozen normalizer stats carried in the checkpoint
    instead of re-running the 500-iter warmup.
    """
    # Ported from refs/dinov3/dinov3/train/train.py:do_train — divergence: no DINO/iBOT/Gram/EMA
    # bookkeeping, no MetricLogger/CombinedDataLoader. We build the optimizer over the student's
    # layer-wise-decayed param groups, run the normalizer warmup (estimate_teacher_statistics) before
    # the loop, then iterate the LVD+ImageNet loader for max_iter steps: apply lr/wd schedule ->
    # zero_grad -> forward_backward (does loss.backward + clip inside backprop_loss) -> optimizer.step.
    model.train()

    # Optimizer over the student's layer-wise-decayed AdamW param groups.
    base_lr = cfg.schedules.lr.peak if "schedules" in cfg else cfg.optim.lr
    base_wd = float(cfg.optim.get("weight_decay", 0.0))
    param_groups = get_params_groups_with_decay(
        model.student,
        lr=float(base_lr),
        wd=base_wd,
        layerwise_decay=float(cfg.optim.get("layerwise_decay", 1.0)),
        patch_embed_lr_mult=float(cfg.optim.get("patch_embed_lr_mult", 0.2)),
    )
    param_groups = fuse_params_groups(param_groups)
    # Adapter heads are trained JOINTLY with the student (paper §4.1) and live OUTSIDE the FSDP
    # student module, so add them as their own AdamW groups at base LR (no layer-wise decay): a
    # weight-decay group for matrices (ndim>=2) and a no-WD group for norms/biases (ndim<=1).
    adapter_decay = [p for p in model.adapters.parameters() if p.requires_grad and p.ndim > 1]
    adapter_no_decay = [p for p in model.adapters.parameters() if p.requires_grad and p.ndim <= 1]
    if adapter_decay:
        param_groups.append({"params": adapter_decay, "lr": float(base_lr),
                             "weight_decay": base_wd, "lr_mult": 1.0, "wd": base_wd})
    if adapter_no_decay:
        param_groups.append({"params": adapter_no_decay, "lr": float(base_lr),
                             "weight_decay": 0.0, "lr_mult": 1.0, "wd": 0.0})
    optimizer = build_optimizer(cfg, param_groups)
    schedulers = build_schedulers(cfg)

    max_iter = _total_iters(cfg)

    # Resume: restore the AdamW moments onto the freshly-built (re-sharded) optimizer. Best-effort —
    # if the FSDP2 re-shard fails for any reason we keep training with fresh moments rather than crash.
    if resume_optimizer_state is not None:
        try:
            if _restore_optimizer_state(model, optimizer, resume_optimizer_state):
                logger.info("Restored optimizer state from checkpoint (resuming at iteration %d)", start_iteration)
        except Exception as e:
            logger.warning("optimizer-state restore failed (%s); continuing with fresh AdamW moments", e)

    # Data loader (LVD-1689M + ImageNet-1k mix; pyramid collate iff crops.global_crops_size is a list).
    data_loader = build_data_loader(cfg, start_iteration=start_iteration)

    # Normalizer warmup: estimate + freeze per-teacher (cls, patch) mean/std before training. On resume
    # we restore the frozen stats from the checkpoint (see main()) and skip the warmup re-estimation.
    if skip_normalizer_init:
        logger.info("Reusing frozen normalizer stats from checkpoint; skipping warmup estimation")
    else:
        model.init_normalizer(data_loader)

    if cfg.multidistillation.enabled:
        global_batch_size = int(cfg.multidistillation.global_batch_size)
    else:
        global_batch_size = int(cfg.train.batch_size_per_gpu) * distributed.get_world_size()

    checkpoint_period = int(cfg.checkpointing.period)

    logger.info("Starting distillation training for %d iterations (from iteration %d)", max_iter, start_iteration)
    iteration = start_iteration
    data_iter = iter(data_loader)
    while iteration < max_iter:
        try:
            data = next(data_iter)
        except StopIteration:
            # The mixed sampler is infinite, but guard against finite single-source loaders.
            data_iter = iter(data_loader)
            data = next(data_iter)

        # Carry the global batch size so MultiDistillationMetaArch can size its subgroup broadcast.
        if isinstance(data, dict):
            data["global_batch_size"] = global_batch_size

        # Learning-rate / weight-decay schedule for this step.
        apply_optim_scheduler(optimizer, schedulers, iteration)

        # Forward + backward (backprop_loss runs an unscaled loss.backward()).
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, iteration=iteration)
        # Grad clip in the loop (matches dinov3): the FSDP2-sharded student (DTensor-aware global norm)
        # and the separately-held adapter heads are each clipped per-unit (see _clip_gradients).
        clip_grad = cfg.optim.get("clip_grad", None)
        if clip_grad:
            _clip_gradients(model, float(clip_grad))
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            loss_val = loss_dict["loss"]
            loss_val = loss_val.item() if torch.is_tensor(loss_val) else float(loss_val)
            logger.info(
                "it=%d  lr=%.3e  wd=%.3e  loss=%.4f",
                iteration,
                schedulers["lr"][iteration],
                schedulers["wd"][iteration],
                loss_val,
            )

        # Periodic checkpointing (student saved under 'teacher.'-prefixed keys).
        if (iteration + 1) % checkpoint_period == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _save_checkpoint(cfg, model, optimizer, iteration)

        iteration += 1

    # Final checkpoint.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _save_checkpoint(cfg, model, optimizer, max_iter - 1)
    logger.info("Distillation training finished at iteration %d", max_iter)


def main(args=None) -> int:
    """Parse args, setup config (or setup_multidistillation), build+parallelize model, do_train."""
    # Ported from refs/dinov3/dinov3/train/train.py:main — divergence: EUPE wraps setup +
    # teardown in eupe.run.init.job_context (which calls setup_job/exit_job for us), routes to
    # MultiDistillationMetaArch iff cfg.multidistillation.enabled (using setup_multidistillation,
    # which itself enables distributed + builds the rank subgroups) else DistillationMetaArch via
    # setup_config, builds the meta-arch on the meta device, FSDP-wraps it with parallelize(), then
    # materializes weights with init_weights() before do_train.
    parsed = get_args_parser().parse_args(args)

    setup_args = EupeSetupArgs(
        config_file=parsed.config_file,
        output_dir=parsed.output_dir,
        opts=list(parsed.opts or []),
    )

    if parsed.multi_distillation:
        logger.info("performing multidistillation run")
        # setup_multidistillation enables distributed + builds the rank subgroups itself, so we run
        # config setup OUTSIDE job_context's distributed.enable (it would double-enable otherwise),
        # and let job_context only manage logging/teardown.
        cfg = setup_multidistillation(setup_args)
        from eupe.train.multidist_meta_arch import MultiDistillationMetaArch

        meta_arch_cls = MultiDistillationMetaArch
        distributed_enabled = False
    else:
        cfg = None
        from eupe.train.distill_meta_arch import DistillationMetaArch

        meta_arch_cls = DistillationMetaArch
        distributed_enabled = True

    with job_context(
        output_dir=parsed.output_dir,
        distributed_enabled=distributed_enabled,
        seed=parsed.seed,
    ):
        if cfg is None:
            cfg = setup_config(setup_args, strict_cfg=False)
        logger.info("Making meta arch %s", meta_arch_cls.__name__)

        # Build the meta-arch. The student backbone is built on the META device by
        # build_model_from_cfg (so the ~1.9B proxy never materializes unsharded); the FROZEN teachers
        # are loaded eagerly with their real checkpoints onto cuda (build_teachers), and the adapter
        # heads / normalizers are built on the default device. We deliberately do NOT wrap this in
        # `torch.device("meta")`: doing so would also put the teachers + adapters on meta, breaking the
        # teacher checkpoint load — and the old blanket model.to_empty() would then zero their weights.
        model = meta_arch_cls(cfg)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resume a preempted SAME-stage run from the latest checkpoint under <output_dir>/ckpt, unless
        # --no-resume. Resume weights take precedence over the Stage-3 init checkpoint (which is only for
        # a fresh start). 390k/100k-iter runs are effectively guaranteed to be preempted on a cluster.
        resume_path, resume_iteration = (None, -1)
        if not parsed.no_resume:
            resume_path, resume_iteration = _find_latest_checkpoint(cfg.train.output_dir)
        resume_payload = None
        if resume_path is not None:
            logger.info("Resuming from checkpoint %s (iteration %d)", resume_path, resume_iteration)
            resume_payload = torch.load(resume_path, map_location="cpu")

        # Stage-3 finetune: an optional student-init checkpoint (the Stage-2 weights) to load BEFORE
        # FSDP sharding, so fully_shard shards the LOADED weights rather than random init. Placeholder
        # "<...>" paths are treated as unset. Students are <=~90M, so the brief unsharded
        # materialization needed to load + then shard is cheap (the 1.9B proxy never takes this path).
        init_ckpt = cfg.student.get("pretrained_weights", None) if "student" in cfg else None
        has_init = bool(init_ckpt) and not str(init_ckpt).startswith("<")

        if resume_payload is not None:
            # Resume: materialize + init the unsharded student, OVERWRITE its params with the resumed
            # weights, THEN FSDP-shard the loaded weights (same ordering as the Stage-3 init path).
            from eupe.models import extract_backbone_state_dict

            if any(p.is_meta for p in model.student.parameters()):
                model.student.to_empty(device=device)
            if hasattr(model.student, "init_weights"):
                model.student.init_weights()
            resume_sd = extract_backbone_state_dict(resume_payload)
            missing, unexpected = model.student.load_state_dict(resume_sd, strict=False)
            if missing or unexpected:
                logger.warning(
                    "resume student load: %d missing / %d unexpected keys", len(missing), len(unexpected)
                )
            model.student = parallelize(model.student, cfg)
        elif has_init:
            # Materialize + init the unsharded student (fills non-persistent RoPE buffers), OVERWRITE
            # its parameters with the checkpoint, THEN FSDP-shard the now-loaded weights.
            if any(p.is_meta for p in model.student.parameters()):
                model.student.to_empty(device=device)
            if hasattr(model.student, "init_weights"):
                model.student.init_weights()
            _load_student_init_checkpoint(model.student, init_ckpt)
            model.student = parallelize(model.student, cfg)
        else:
            # FSDP-wrap ONLY the student backbone (teachers stay frozen/eval inside the meta-arch).
            model.student = parallelize(model.student, cfg)
            # Materialize + initialize the student if it is still on meta (e.g. the single-process
            # fallback where parallelize skipped FSDP). We never to_empty the WHOLE model — that would
            # wipe the teachers' loaded checkpoints and the adapter init.
            if any(p.is_meta for p in model.student.parameters()):
                model.student.to_empty(device=device)
            if hasattr(model.student, "init_weights"):
                model.student.init_weights()  # FSDP2 meta-init path — validate on GPU.

        # Place the trainable adapters + frozen normalizer buffers on the compute device (the teachers
        # were already moved to cuda by build_teachers).
        if torch.cuda.is_available():
            model.adapters.cuda()
            model.normalizers.cuda()

        # Resume: restore the frozen normalizer stats (skip the 500-iter warmup) when the checkpoint
        # carries them; older checkpoints without them fall back to re-estimating in do_train.
        skip_normalizer_init = False
        if resume_payload is not None and resume_payload.get("normalizers"):
            model.normalizers.load_state_dict(resume_payload["normalizers"])
            skip_normalizer_init = True

        do_train(
            cfg,
            model,
            resume_optimizer_state=(resume_payload.get("optimizer") if resume_payload is not None else None),
            start_iteration=(resume_iteration + 1 if resume_payload is not None else 0),
            skip_normalizer_init=skip_normalizer_init,
        )
    return 0


if __name__ == "__main__":
    main()
