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

import pytest
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


def test_named_factory_used_when_only_ffn_ratio_present():
    # P1 regression: ssl_default supplies ffn_ratio (4.0) but leaves the structural dims to the named
    # factory. ffn_ratio alone must NOT trigger the explicit-dims constructor, else the factory's dims
    # (here vit_small's 384/12/6) are silently replaced by DinoVisionTransformer's 768/12/12 defaults.
    default = OmegaConf.load(_DEFAULT_CFG).student
    cfg = OmegaConf.merge(default, OmegaConf.create({"arch": "vit_small", "n_storage_tokens": 0}))
    for k in ("embed_dim", "depth", "num_heads"):  # precondition: NO structural dims in the cfg
        assert cfg.get(k) is None
    assert cfg.get("ffn_ratio") is not None  # the landmine: the default always supplies ffn_ratio
    student, _teacher, dim = build_model(cfg, img_size=64, device=None)
    assert dim == 384 and student.embed_dim == 384  # vit_small factory dims, NOT the 768 DinoViT default
    assert student.n_blocks == 12 and student.num_heads == 6


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


def test_rope_dtype_is_threaded_from_cfg():
    # build_model must thread pos_embed_rope_dtype so the YAML (every ViT config sets fp32) is
    # authoritative; it previously fell back to DinoVisionTransformer's bf16 default while the hub/eval
    # builder passes fp32 — a silent train/eval RoPE-precision mismatch.
    tiny = dict(arch="vit_small", embed_dim=64, depth=1, num_heads=2, n_storage_tokens=0)
    s_fp32, _, _ = build_model(_student_cfg("students/vitt_p16.yaml", pos_embed_rope_dtype="fp32", **tiny),
                               img_size=32, device=None)
    s_bf16, _, _ = build_model(_student_cfg("students/vitt_p16.yaml", pos_embed_rope_dtype="bf16", **tiny),
                               img_size=32, device=None)
    assert s_fp32.rope_embed.periods.dtype == torch.float32
    assert s_bf16.rope_embed.periods.dtype == torch.bfloat16


def test_proxy_teacher_builds_from_base_keyless_config(tmp_path):
    # CRITICAL regression: ProxyTeacher loads the proxy YAML standalone (no ssl_default merge). Like
    # vitg_p16.yaml, this config omits base-only keys (qkv_bias, the rope min/max/shift/jitter periods,
    # the untie-norm flags). build_model dereferences them unconditionally, so without ProxyTeacher's
    # default-merge this raised ConfigAttributeError and crashed every Stage-2/3 run.
    import omegaconf
    from omegaconf import OmegaConf

    from eupe.distill.teachers import ProxyTeacher

    # A tiny proxy-shaped config MISSING the base keys (mirrors vitg_p16.yaml's key set, small dims).
    cfg = OmegaConf.create({"student": {
        "arch": "vit_small", "embed_dim": 64, "depth": 2, "num_heads": 2, "ffn_ratio": 4.0,
        "patch_size": 16, "n_storage_tokens": 4, "layerscale": 1.0e-05, "norm_layer": "layernormbf16",
        "ffn_layer": "mlp", "mask_k_bias": True, "pos_embed_type": "rope", "pos_embed_rope_base": 100,
        "pos_embed_rope_normalize_coords": "separate", "pos_embed_rope_rescale_coords": 2,
        "pos_embed_rope_dtype": "fp32",
    }})
    proxy_yaml = tmp_path / "proxy_tiny.yaml"
    OmegaConf.save(cfg, proxy_yaml)

    # Sanity: the raw (un-merged) config really does crash build_model — i.e. the bug is real.
    with pytest.raises(omegaconf.errors.ConfigAttributeError):
        build_model(cfg.student, only_teacher=True, img_size=32)

    # The fix: ProxyTeacher merges ssl_default's student block first, so it builds. Placeholder
    # checkpoint ("<...>") skips the load and just init_weights() on CPU.
    teacher = ProxyTeacher(config=str(proxy_yaml), checkpoint="<none>", native_resolution=32)
    assert teacher.embed_dim == 64
    assert teacher.native_resolution == 32


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
