# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""FSDP sharding, activation checkpointing, and compile helpers."""

from .ac_compile_parallelize import (
    apply_activation_checkpointing,
    apply_compile,
    parallelize,
)

__all__ = ["parallelize", "apply_activation_checkpointing", "apply_compile"]
