# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Distillation data: LVD-1689M + ImageNet-1k mix and the Stage-3 resolution pyramid (paper §3.4, §4.1).

Interleave homogeneous ImageNet-1k batches with heterogeneous LVD-1689M batches, P(ImageNet)=0.10.
Augmentations: random-resized crop, horizontal flip, color jitter, Gaussian blur, solarization
(Stage 2). Stage 3 adds an independent per-sample scale draw from {256,384,512}. Reuses
eupe/data/{loaders.py, samplers.py, transforms.py}.
"""
import logging
from typing import Iterable, List

logger = logging.getLogger("eupe")


class MixedSampler:
    """Yield ImageNet-1k batches with probability `imagenet_prob`, else LVD-1689M batches.

    Args:
        lvd_dataset, imagenet_dataset: the two sources.
        imagenet_prob: 0.10 (paper §3.4).
    """

    def __init__(self, lvd_dataset, imagenet_dataset, imagenet_prob: float = 0.10):
        raise NotImplementedError("TODO: store sources + Bernoulli(imagenet_prob) batch routing")

    def __iter__(self):
        raise NotImplementedError("TODO: yield homogeneous/heterogeneous batches by probability")


def build_pyramid_collate(scales: List[int]):
    """Return a collate_fn that resizes each sample to an independently sampled scale (Stage 3)."""
    raise NotImplementedError("TODO: per-sample random scale collate for {256,384,512}")


def make_distillation_data_loader(cfg) -> Iterable:
    """Build the mixed sampler + transforms + DataLoader from cfg (crops, batch_size_per_gpu, workers).

    If cfg.crops.global_crops_size is a list, attach build_pyramid_collate (Stage 3).
    """
    raise NotImplementedError("TODO: assemble datasets, MixedSampler, transforms, DataLoader")
