# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Distillation data: LVD-1689M + ImageNet-1k mix and the Stage-3 resolution pyramid (paper §3.4, §4.1).

Interleave homogeneous ImageNet-1k batches with heterogeneous LVD-1689M batches, P(ImageNet)=0.10.
Augmentations: random-resized crop, horizontal flip, color jitter, Gaussian blur, solarization
(Stage 2). For Stage 3 the loader produces crops at the LARGEST pyramid size; the per-iteration
pyramid scaling (teacher and student each pick a scale INDEPENDENTLY, paper §3.1) is done in
eupe/train/multidist_meta_arch.py so the two scales are rank-synchronized for the cross-rank
all-gather. Reuses eupe/data/{loaders.py, samplers.py, transforms.py}.
"""
import logging
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .loaders import make_dataset  # dataset-string parsing + dataset construction
from .samplers import InfiniteSampler  # rank-aware infinite index streams
from .transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("eupe")


def make_distillation_transform(
    *,
    global_crops_size: int,
    global_crops_scale=(0.32, 1.0),
    horizontal_flips: bool = True,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    """Single-view distillation augmentation (paper §4.1 / report §6.1).

    RandomResizedCrop -> horizontal flip -> color jitter -> Gaussian blur -> solarization -> normalize.
    """
    # Ported from refs/dinov3/dinov3/data/augmentations.py:DataAugmentationDINO — single global view
    # (no local crops / no multi-crop), distillation only matches one student view per image.
    geometric = v2.Compose(
        [
            v2.RandomResizedCrop(
                global_crops_size,
                scale=tuple(global_crops_scale),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
        ]
    )
    color_jittering = v2.Compose(
        [
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            v2.RandomGrayscale(p=0.2),
        ]
    )
    photometric = v2.Compose(
        [
            color_jittering,
            GaussianBlur(p=0.5),
            v2.RandomSolarize(threshold=128, p=0.2),
        ]
    )
    normalize = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    transform = v2.Compose([geometric, photometric, normalize])
    logger.info(f"Built distillation train transform\n{transform}")
    return transform


class MixedSampler:
    """Yield ImageNet-1k batches with probability `imagenet_prob`, else LVD-1689M batches.

    Each draw is a Bernoulli(`imagenet_prob`) coin flip per *batch*: heads -> a homogeneous batch
    of indices into ImageNet-1k, tails -> a heterogeneous batch of indices into LVD-1689M. The
    underlying per-source index streams come from the rank-aware infinite samplers in
    eupe/data/samplers.py, so the stream never terminates and is correctly sharded across ranks.

    Each yielded element is a list (length `batch_size`) of `(source, index)` tuples, where
    `source` is "imagenet" or "lvd". Pairing this with a `DataLoader(batch_sampler=...)` over a
    ConcatDataset-style routing dataset lets a single loader dispatch to the right source.

    Args:
        lvd_dataset, imagenet_dataset: the two sources.
        imagenet_prob: 0.10 (paper §3.4).
    """

    def __init__(
        self,
        lvd_dataset,
        imagenet_dataset,
        imagenet_prob: float = 0.10,
        *,
        batch_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        start_iteration: int = 0,
    ):
        # Reuse eupe's rank-aware infinite index streams (one per source) so the mixed stream is
        # both infinite and correctly sharded; the per-batch Bernoulli only chooses *which* stream.
        self.lvd_dataset = lvd_dataset
        self.imagenet_dataset = imagenet_dataset
        self.imagenet_prob = float(imagenet_prob)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = shuffle
        # Resume: fast-forward this many batches at __iter__ time so a restarted run continues the data
        # stream where it left off (routing generator + both per-source index streams advance together)
        # instead of replaying from batch 0.
        self.start_iteration = int(start_iteration)
        self._lvd_sampler = InfiniteSampler(
            sample_count=len(lvd_dataset), shuffle=shuffle, seed=seed
        )
        # Offset the imagenet seed so the two streams are decorrelated.
        self._imagenet_sampler = InfiniteSampler(
            sample_count=len(imagenet_dataset), shuffle=shuffle, seed=seed + 1
        )

    def __iter__(self):
        # Generator seeded per-rank-consistently so the Bernoulli routing is identical on all ranks
        # (every rank must agree on whether a given batch is IN1k or LVD to keep batches homogeneous).
        generator = torch.Generator().manual_seed(self.seed)
        lvd_it = iter(self._lvd_sampler)
        imagenet_it = iter(self._imagenet_sampler)
        # Resume fast-forward: replay the deterministic routing + index draws for the already-consumed
        # batches WITHOUT yielding, so both per-source streams and the routing generator land exactly
        # where the preempted run stopped.
        for _ in range(self.start_iteration):
            use_imagenet = torch.rand(1, generator=generator).item() < self.imagenet_prob
            index_it = imagenet_it if use_imagenet else lvd_it
            for _ in range(self.batch_size):
                next(index_it)
        while True:
            use_imagenet = (
                torch.rand(1, generator=generator).item() < self.imagenet_prob
            )
            if use_imagenet:
                source = "imagenet"
                index_it = imagenet_it
            else:
                source = "lvd"
                index_it = lvd_it
            batch = [(source, next(index_it)) for _ in range(self.batch_size)]
            yield batch


class _MixedRoutingDataset(torch.utils.data.Dataset):
    """Map `(source, index)` keys (from MixedSampler) to samples from the matching source."""

    def __init__(self, lvd_dataset, imagenet_dataset):
        self._sources = {"lvd": lvd_dataset, "imagenet": imagenet_dataset}

    def __getitem__(self, key):
        source, index = key
        image, _target = self._sources[source][index]
        return image, source


def _default_collate(batch):
    """Stack a homogeneous-size batch of (image, source) into ([B,C,H,W], [source,...])."""
    images, sources = zip(*batch)
    return torch.stack(list(images), dim=0), list(sources)


def make_distillation_data_loader(cfg, start_iteration: int = 0) -> Iterable:
    """Build the mixed sampler + transforms + DataLoader from cfg (crops, batch_size_per_gpu, workers).

    `cfg.train.dataset_path` holds the two sources joined by "+": "<LVD>+<IN1k>" (each is a
    `make_dataset` string, e.g. "ImageNet:split=TRAIN:root=..."). The single-source case (no "+")
    is treated as both LVD and ImageNet pointing at the same dataset.

    If `cfg.crops.global_crops_size` is a list (Stage 3), build the RRC transform at the LARGEST
    scale and stack batches at that size; the per-iteration pyramid scaling (independent teacher /
    student scales) is handled in the meta-arch. Otherwise use the scalar crop size.
    """
    # Provenance: reuses eupe/data/{loaders,samplers,transforms}.py; mixing recipe (P(IN1k)=0.10)
    # ported from refs/dinov3 IN1k/LVD interleave — single-view distillation augmentation.
    global_crops_size = cfg.crops.global_crops_size
    is_pyramid = isinstance(global_crops_size, (list, tuple)) or (
        hasattr(global_crops_size, "__iter__") and not isinstance(global_crops_size, (str, bytes, int))
    )
    if is_pyramid:
        scales = [int(s) for s in global_crops_size]
        crop_size = max(scales)
    else:
        scales = None
        crop_size = int(global_crops_size)

    # Optional crop-scale / flip knobs (fall back to distillation defaults if absent).
    global_crops_scale = getattr(cfg.crops, "global_crops_scale", (0.32, 1.0))
    horizontal_flips = getattr(cfg.crops, "horizontal_flips", True)
    mean = getattr(cfg.crops, "rgb_mean", IMAGENET_DEFAULT_MEAN)
    std = getattr(cfg.crops, "rgb_std", IMAGENET_DEFAULT_STD)

    transform = make_distillation_transform(
        global_crops_size=crop_size,
        global_crops_scale=global_crops_scale,
        horizontal_flips=horizontal_flips,
        mean=mean,
        std=std,
    )

    dataset_path = cfg.train.dataset_path
    if "+" in dataset_path:
        lvd_str, imagenet_str = (tok.strip() for tok in dataset_path.split("+", 1))
    else:
        lvd_str = imagenet_str = dataset_path.strip()
        logger.warning(
            "cfg.train.dataset_path has no '+' separator: using the SAME dataset for both the LVD and "
            "IN1k sources. The paper's LVD-1689M + ImageNet-1k mixture (§3.4) is therefore NOT in effect "
            "— set dataset_path='<LVD_source>+<ImageNet:...>' for a faithful run. (%s)",
            dataset_path,
        )

    lvd_dataset = make_dataset(dataset_str=lvd_str, transform=transform)
    imagenet_dataset = make_dataset(dataset_str=imagenet_str, transform=transform)

    batch_size = int(cfg.train.batch_size_per_gpu)
    num_workers = int(cfg.train.num_workers)
    seed = int(getattr(cfg.train, "seed", 0))

    batch_sampler = MixedSampler(
        lvd_dataset,
        imagenet_dataset,
        imagenet_prob=getattr(getattr(cfg, "distill", cfg), "imagenet_prob", 0.10),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        start_iteration=start_iteration,
    )
    routing_dataset = _MixedRoutingDataset(lvd_dataset, imagenet_dataset)

    # Stage 3 produces uniform max-scale crops here; multidist_meta_arch resizes to the independent
    # per-iteration teacher/student pyramid scales (rank-synchronized). So a plain stack collate is
    # used for both fixed-res and pyramid configs.
    collate_fn = _default_collate

    logger.info("using PyTorch data loader (distillation mixed LVD+IN1k)")
    data_loader = DataLoader(
        routing_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_loader
