# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Guards the Stage-2/3 learning-rate recipe (M17/M24).

The paper uses a single flat base LR for the distillation students; ssl_default's layerwise_decay=0.9
and patch_embed_lr_mult=0.2 must NOT leak onto them. With both disabled (1.0), every param group's
lr_mult is 1.0; the second test confirms the assertion is meaningful (decay enabled -> non-uniform).
"""
import os

from omegaconf import OmegaConf

from eupe.models import build_model
from eupe.train.param_groups import get_params_groups_with_decay

_HERE = os.path.dirname(__file__)
_TRAIN_CFG = os.path.normpath(os.path.join(_HERE, "..", "..", "eupe", "configs", "train"))
_DEFAULT_CFG = os.path.normpath(os.path.join(_HERE, "..", "..", "eupe", "configs", "ssl_default_config.yaml"))


def _vit_tiny_student():
    default = OmegaConf.load(_DEFAULT_CFG).student
    node = OmegaConf.load(os.path.join(_TRAIN_CFG, "students/vitt_p16.yaml")).student
    student, _teacher, _dim = build_model(OmegaConf.merge(default, node), img_size=64, device=None)
    return student


def test_flat_lr_when_layerwise_decay_disabled():
    # M17/M24: Stage 2/3 set layerwise_decay=1.0 and patch_embed_lr_mult=1.0 -> all groups flat.
    groups = get_params_groups_with_decay(
        _vit_tiny_student(), lr=2e-5, wd=1e-4, layerwise_decay=1.0, patch_embed_lr_mult=1.0
    )
    assert groups, "expected non-empty param groups"
    assert all(abs(g["lr_mult"] - 1.0) < 1e-9 for g in groups)


def test_layerwise_decay_is_nonuniform_when_enabled():
    # Sanity check that the test above is meaningful: ssl_default's 0.9 / 0.2 DO vary the lr_mult,
    # which is exactly what the Stage-2/3 override now suppresses.
    groups = get_params_groups_with_decay(
        _vit_tiny_student(), lr=2e-5, wd=1e-4, layerwise_decay=0.9, patch_embed_lr_mult=0.2
    )
    assert any(abs(g["lr_mult"] - 1.0) > 1e-6 for g in groups)
