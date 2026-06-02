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
        # Ported from refs/dinov3/dinov3/train/cosine_lr_scheduler.py:CosineScheduler.__init__
        # — dropped the trunc_extra branch (not in frozen interface); schedule is
        #   freeze (start_warmup_value) -> linear warmup -> cosine decay, concatenated to total_iters.
        self.final_value = np.float64(final_value)
        self.total_iters = total_iters

        freeze_schedule = np.full((freeze_iters,), start_warmup_value, dtype=np.float64)

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, cosine_schedule), dtype=np.float64)

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it: int) -> float:
        return self.schedule[min(it, len(self.schedule) - 1)]
