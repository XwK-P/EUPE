# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Regression tests for build_model architecture dispatch + checkpoint extraction.

Guards:
  * C1 — ViT-T honors its explicit 192/12/3 dims (build_model previously ignored them and built
    vit_small's 384/12/6).
  * C2 — ConvNeXt students build at all (build_model previously raised NotImplementedError).
  * H8 — ffn_ratio is threaded so the proxy can be widened toward ~1.9B.
  * checkpoint round-trip — extract_backbone_state_dict recovers plain keys from the training payload.
"""
import os

import torch
from omegaconf import OmegaConf

from eupe.models import build_model, extract_backbone_state_dict

_HERE = os.path.dirname(__file__)
_TRAIN_CFG = os.path.normpath(os.path.join(_HERE, "..", "..", "eupe", "configs", "train"))
_DEFAULT_CFG = os.path.normpath(os.path.join(_HERE, "..", "..", "eupe", "configs", "ssl_default_config.yaml"))


def _student_cfg(rel_yaml, **overrides):
    """Merge ssl_default's student defaults with a shipped student yaml (+ optional overrides)."""
    default = OmegaConf.load(_DEFAULT_CFG).student
    node = OmegaConf.load(os.path.join(_TRAIN_CFG, rel_yaml)).student
    merged = OmegaConf.merge(default, node)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(overrides))
    return merged


def test_vit_tiny_honors_explicit_dims():
    # C1: arch=vit_small but the explicit 192/12/3 must win (ViT-T 6M, NOT vit_small's 384/12/6).
    student, _teacher, dim = build_model(_student_cfg("students/vitt_p16.yaml"), img_size=64, device=None)
    assert dim == 192
    assert student.embed_dim == 192
    assert student.n_blocks == 12
    assert student.num_heads == 3


def test_ffn_ratio_is_threaded():
    # H8 mechanism: build_model must thread ffn_ratio (so the proxy can be widened toward ~1.9B).
    tiny = dict(arch="vit_small", embed_dim=64, depth=1, num_heads=2, n_storage_tokens=0)
    s4, _, _ = build_model(_student_cfg("students/vitt_p16.yaml", ffn_ratio=4.0, **tiny), img_size=32, device=None)
    s8, _, _ = build_model(_student_cfg("students/vitt_p16.yaml", ffn_ratio=8.0, **tiny), img_size=32, device=None)
    assert sum(p.numel() for p in s8.parameters()) > sum(p.numel() for p in s4.parameters())


def test_proxy_config_widens_ffn_toward_1p9b():
    # H8: the shipped proxy config must not be the stock ~1.1B giant2 (ffn_ratio 4).
    cfg = OmegaConf.load(os.path.join(_TRAIN_CFG, "proxy", "vitg_p16.yaml")).student
    assert float(cfg.ffn_ratio) > 4.0
    assert int(cfg.embed_dim) == 1536 and int(cfg.depth) == 40


def test_convnext_students_build_and_emit_tokens():
    # C2: ConvNeXt students must build (previously raised NotImplementedError) and emit the
    # {x_norm_clstoken, x_norm_patchtokens} dict the distillation meta-arch consumes.
    for rel, edim in (("students/convnext_tiny_p16.yaml", 768), ("students/convnext_base_p16.yaml", 1024)):
        student, _teacher, dim = build_model(_student_cfg(rel), img_size=64, device=None)
        assert dim == edim and student.embed_dim == edim
        out = student.forward_features(torch.randn(1, 3, 64, 64))
        assert out["x_norm_clstoken"].shape == (1, edim)
        assert out["x_norm_patchtokens"].shape[-1] == edim


def test_extract_backbone_state_dict_unwraps_training_payload():
    # Guards the checkpoint round-trip fix: _save_checkpoint writes {"teacher": {"teacher.<k>": v}}
    # with possible FSDP/AC/compile name decorations; eval/proxy loaders must recover plain keys.
    payload = {
        "teacher": {
            "teacher.cls_token": torch.zeros(1),
            "teacher.blocks.0._checkpoint_wrapped_module.norm.weight": torch.zeros(2),
            "teacher._orig_mod.head.bias": torch.zeros(3),
        },
        "optimizer": {"x": 1},
        "iteration": 5,
    }
    sd = extract_backbone_state_dict(payload)
    assert set(sd.keys()) == {"cls_token", "blocks.0.norm.weight", "head.bias"}
