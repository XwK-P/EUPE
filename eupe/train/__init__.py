# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""EUPE distillation trainer (DINOv3-style) for the multi-stage pipeline."""

from .distill_meta_arch import DistillationMetaArch
from .multidist_meta_arch import MultiDistillationMetaArch

__all__ = ["DistillationMetaArch", "MultiDistillationMetaArch"]
