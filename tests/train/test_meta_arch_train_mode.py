# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Regression test for the DistillationMetaArch.train(mode) contract.

Guard: nn.Module.eval() is implemented as self.train(False), so the meta-arch's train() override MUST
accept `mode`; a no-arg override crashes every eval/validation/export path with a TypeError. The
override must also keep the frozen teachers in eval() regardless of mode (deterministic targets).

DistillationMetaArch.__init__ builds students + loads real teacher checkpoints (too heavy for a unit
test), so we construct a bare instance via __new__ and register only the `teachers` submodule the
train() override touches, then exercise the real method.
"""
import torch
from torch import nn

from eupe.train.distill_meta_arch import DistillationMetaArch


def _bare_meta_arch():
    obj = DistillationMetaArch.__new__(DistillationMetaArch)
    nn.Module.__init__(obj)
    obj.student = nn.Linear(2, 2)  # a trainable submodule that should follow train/eval
    obj.teachers = nn.ModuleDict({"t": nn.Linear(2, 2)})  # frozen targets: always eval
    return obj


def test_eval_does_not_crash_and_forces_eval_mode():
    obj = _bare_meta_arch()
    obj.eval()  # nn.Module.eval() -> self.train(False); a no-arg override would raise TypeError here
    assert obj.training is False
    assert obj.student.training is False
    assert obj.teachers["t"].training is False


def test_train_default_trains_student_but_keeps_teachers_eval():
    obj = _bare_meta_arch()
    ret = obj.train()  # default mode=True
    assert ret is obj  # train() returns self
    assert obj.training is True
    assert obj.student.training is True
    assert obj.teachers["t"].training is False  # teachers forced back to eval


def test_train_false_matches_eval():
    obj = _bare_meta_arch()
    obj.train(False)
    assert obj.training is False
    assert obj.teachers["t"].training is False
