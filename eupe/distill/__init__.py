# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""EUPE distillation objective: teachers, adapters, normalization, loss."""

from .adapters import AdapterHead, AdapterHeadSet
from .loss import DistillationLoss
from .normalize import FeatureNormalizer, estimate_teacher_statistics
from .teachers import TeacherModel, build_teachers

__all__ = [
    "TeacherModel",
    "build_teachers",
    "AdapterHead",
    "AdapterHeadSet",
    "FeatureNormalizer",
    "estimate_teacher_statistics",
    "DistillationLoss",
]
