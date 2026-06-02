# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Regression tests for checkpoint discovery + optimizer-state resume (single-process path).

Guards:
  * _find_latest_checkpoint picks the highest-numbered training_<it>.pth (numeric, not lexicographic).
  * the optimizer state round-trips through _consolidated_optimizer_state / _restore_optimizer_state
    so a resumed run continues with the prior AdamW moments (the distributed FSDP2 path uses
    torch.distributed.checkpoint and is exercised on a real multi-rank run).
  * _save_checkpoint persists the jointly-trained adapter heads so a resumed run does NOT continue
    with randomly re-initialized adapters driven by the restored (trained-adapter) AdamW moments.
"""
import types

import torch

from eupe.train.train import (
    _consolidated_optimizer_state,
    _find_latest_checkpoint,
    _restore_optimizer_state,
    _save_checkpoint,
)


def test_find_latest_checkpoint_picks_highest_iteration(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for it in (49, 1250, 389999, 100):
        (ckpt / f"training_{it}.pth").write_bytes(b"x")
    (ckpt / "training_notanint.pth").write_bytes(b"x")  # malformed name must be skipped, not crash
    path, it = _find_latest_checkpoint(str(tmp_path))
    assert it == 389999
    assert path.name == "training_389999.pth"  # 389999 > 1250 numerically (lexicographic would pick 49/1250)


def test_find_latest_checkpoint_missing_dir(tmp_path):
    path, it = _find_latest_checkpoint(str(tmp_path / "does_not_exist"))
    assert path is None and it == -1


def test_optimizer_state_roundtrips_single_process():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()  # populate exp_avg / exp_avg_sq moments

    state = _consolidated_optimizer_state(model, opt)  # single-process -> plain optimizer.state_dict()

    model2 = torch.nn.Linear(4, 4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    assert _restore_optimizer_state(model2, opt2, state) is True

    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()
    torch.testing.assert_close(s1[0]["exp_avg"], s2[0]["exp_avg"])
    torch.testing.assert_close(s1[0]["exp_avg_sq"], s2[0]["exp_avg_sq"])


def test_restore_optimizer_state_none_is_noop():
    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-3)
    assert _restore_optimizer_state(None, opt, None) is False  # missing optimizer state -> fresh moments


class _Bag(torch.nn.Module):
    """Minimal stand-in for a meta-arch submodule (student / normalizers / adapters)."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


class _FakeMetaArch:
    """Just the attributes _save_checkpoint touches: student / normalizers / adapters."""

    def __init__(self):
        self.student = _Bag()
        self.normalizers = _Bag()
        self.adapters = _Bag()


def test_save_checkpoint_persists_adapter_weights(tmp_path):
    # Regression: the optimizer state carries the adapters' AdamW moments, so the checkpoint MUST also
    # carry the adapter weights. Otherwise a resumed run loads moments tuned for the trained heads onto
    # randomly re-initialized heads (mismatched projection -> corrupted gradients after every preempt).
    torch.manual_seed(0)
    model = _FakeMetaArch()
    # Drive the adapter weights away from their init so a fresh module is guaranteed to differ.
    with torch.no_grad():
        model.adapters.lin.weight.add_(1.0)
        model.adapters.lin.bias.add_(1.0)
    opt = torch.optim.AdamW(model.adapters.parameters(), lr=1e-3)
    cfg = types.SimpleNamespace(train=types.SimpleNamespace(output_dir=str(tmp_path)))

    _save_checkpoint(cfg, model, opt, iteration=5)

    payload = torch.load(tmp_path / "ckpt" / "training_5.pth", map_location="cpu")
    assert "adapters" in payload, "checkpoint payload must include adapter weights"

    saved = payload["adapters"]
    orig = model.adapters.state_dict()
    assert saved.keys() == orig.keys()
    for k in orig:
        torch.testing.assert_close(saved[k], orig[k])

    # Round-trip: a freshly-initialized adapter module (random init) loads back to the trained weights.
    fresh = _Bag()
    assert not torch.allclose(fresh.lin.weight, model.adapters.lin.weight)  # fresh init differs pre-load
    missing, unexpected = fresh.load_state_dict(payload["adapters"], strict=False)
    assert not missing and not unexpected
    torch.testing.assert_close(fresh.lin.weight, model.adapters.lin.weight)
    torch.testing.assert_close(fresh.lin.bias, model.adapters.lin.bias)
