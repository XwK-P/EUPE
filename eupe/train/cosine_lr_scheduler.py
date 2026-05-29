# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Cosine schedules for LR / weight-decay / teacher momentum (mirrors dinov3).

Supports schedules-v2 (set peak directly) to bypass the implicit sqrt_wrt_1024 LR scaling in
eupe/configs/config.py::apply_scaling_rules_to_cfg, so reproducers hit the paper's 2e-5 (Stage 2)
/ 1e-5 (Stage 3) regardless of GPU count.
"""
import numpy as np


class CosineScheduler:
    """Per-iteration schedule value via __getitem__(iteration).

    Args:
        base_value, final_value: endpoints.
        total_iters: schedule length.
        warmup_iters: linear warmup length.
        start_warmup_value: warmup start.
        freeze_iters: hold start_warmup_value for this many iters first.
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
    ):
        # TODO: precompute self.schedule as a numpy array of length total_iters
        # (freeze -> linear warmup -> cosine decay). See dinov3/train/cosine_lr_scheduler.py.
        raise NotImplementedError("TODO: precompute freeze+warmup+cosine schedule array")

    def __getitem__(self, it: int) -> float:
        raise NotImplementedError("TODO: return self.schedule[min(it, len-1)]")
