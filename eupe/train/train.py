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
