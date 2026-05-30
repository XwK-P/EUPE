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


def build_data_loader(cfg):
    """eupe.data.distillation_loaders.make_distillation_data_loader(cfg)."""
    return make_distillation_data_loader(cfg)


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


def _save_checkpoint(cfg, model, optimizer, iteration: int) -> None:
    """Save the trained student under 'teacher.'-prefixed keys (subgroup-main process only)."""
    # Ported from refs/dinov3/dinov3/train/train.py:do_train checkpointing + do_test teacher save —
    # divergence: EUPE has no dinov3 checkpointer; we save a plain consolidated dict {"teacher": ...,
    # "optimizer": ..., "iteration": ...} where the model weights are the trained STUDENT re-keyed
    # under 'teacher.' so eupe.models.build_model_for_eval / ProxyTeacher can reload it directly.
    if distributed.is_enabled() and not distributed.is_subgroup_main_process():
        return
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"training_{iteration}.pth"
    payload = {
        "teacher": _state_dict_with_teacher_prefix(model.student),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, ckpt_path)
    logger.info("Saved checkpoint: %s", ckpt_path)


def _clip_gradients(model, max_norm: float) -> None:
    """Clip grads of the (FSDP-sharded) student and the (non-FSDP) adapter heads.

    FSDP1 wraps the student and exposes a ``clip_grad_norm_`` that reduces the *global* sharded
    norm; a plain ``nn.Module`` (single-process fallback) has no such method, so the free clip is
    used there. The adapter heads live outside the FSDP unit and are always clipped with the free clip.
    """
    student = model.student
    if hasattr(student, "clip_grad_norm_"):
        student.clip_grad_norm_(max_norm)  # FSDP-aware global-norm clip
    else:
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
    adapter_grads = [p for p in model.adapters.parameters() if p.grad is not None]
    if adapter_grads:
        torch.nn.utils.clip_grad_norm_(adapter_grads, max_norm)


def do_train(cfg, model) -> None:
    """Main loop: init_normalizer -> for it in range(max_iter): forward_backward; schedule; checkpoint."""
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

    # Data loader (LVD-1689M + ImageNet-1k mix; pyramid collate iff crops.global_crops_size is a list).
    data_loader = build_data_loader(cfg)

    # Normalizer warmup: estimate + freeze per-teacher (cls, patch) mean/std before training.
    model.init_normalizer(data_loader)

    if cfg.multidistillation.enabled:
        global_batch_size = int(cfg.multidistillation.global_batch_size)
    else:
        global_batch_size = int(cfg.train.batch_size_per_gpu) * distributed.get_world_size()

    checkpoint_period = int(cfg.checkpointing.period)

    logger.info("Starting distillation training for %d iterations", max_iter)
    iteration = 0
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
        # Grad clip in the loop (matches dinov3) so it spans the FSDP-sharded student AND the
        # separately-held adapter heads with the correct global norm.
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

        # FSDP-wrap ONLY the student backbone (teachers stay frozen/eval inside the meta-arch).
        model.student = parallelize(model.student, cfg)

        # Materialize + initialize the student if it is still on meta (e.g. the single-process
        # fallback where parallelize skipped FSDP). We never to_empty the WHOLE model — that would
        # wipe the teachers' loaded checkpoints and the adapter init.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if any(p.is_meta for p in model.student.parameters()):
            model.student.to_empty(device=device)
        if hasattr(model.student, "init_weights"):
            model.student.init_weights()  # NOTE: FSDP1 meta-init path — validate on GPU.
        # Place the trainable adapters + frozen normalizer buffers on the compute device (the teachers
        # were already moved to cuda by build_teachers).
        if torch.cuda.is_available():
            model.adapters.cuda()
            model.normalizers.cuda()

        do_train(cfg, model)
    return 0


if __name__ == "__main__":
    main()
