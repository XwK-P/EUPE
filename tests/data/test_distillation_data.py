# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Regression tests for the distillation data pipeline.

Guards:
  * ImageFolder + the LVD1689M alias — the proprietary LVD source resolves to a label-free folder
    reader so the shipped `<LVD1689M:root=/PATH>` config slot is buildable (paper §3.4 mixture).
  * MixedSampler resume fast-forward — a restarted run continues the data stream instead of replaying.
"""
import itertools

from PIL import Image

from eupe.data.distillation_loaders import MixedSampler
from eupe.data.loaders import make_dataset


def test_image_folder_resolves_via_lvd_alias(tmp_path):
    root = tmp_path / "imgs" / "shard0"
    root.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (8, 8), (i * 30, 0, 0)).save(root / f"{i}.png")
    # 'LVD1689M' and 'ImageFolder' must both resolve to the recursive folder reader.
    for name in ("LVD1689M", "ImageFolder"):
        ds = make_dataset(dataset_str=f"{name}:root={tmp_path / 'imgs'}")
        assert len(ds) == 4
        image, target = ds[0]
        assert isinstance(image, Image.Image)
        assert target == 0  # label-free


class _DS:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_mixed_sampler_resume_matches_skipped_fresh_stream():
    # start_iteration=k must reproduce exactly the fresh stream advanced by k batches (routing generator
    # + both per-source index streams advance in lockstep), so resume continues rather than replays.
    def mk(start):
        return MixedSampler(_DS(100), _DS(100), imagenet_prob=0.3, batch_size=2, seed=7, start_iteration=start)

    fresh = list(itertools.islice(iter(mk(0)), 5))
    resumed = list(itertools.islice(iter(mk(3)), 2))
    assert resumed == fresh[3:5]


def test_mixed_sampler_batches_are_source_homogeneous():
    # Each batch is all-IN1k or all-LVD (paper §3.4: homogeneous IN1k / heterogeneous LVD batches).
    sampler = MixedSampler(_DS(50), _DS(50), imagenet_prob=0.5, batch_size=4, seed=0)
    for batch in itertools.islice(iter(sampler), 20):
        sources = {src for src, _idx in batch}
        assert len(sources) == 1
