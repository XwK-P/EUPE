# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

"""Generic recursive image-folder dataset for label-free distillation sources.

The EUPE recipe (§3.4) trains on LVD-1689M + ImageNet-1k. LVD-1689M is a proprietary DINOv3 web
corpus with no public loader, so this ImageFolder adapter lets a reproducer point the heterogeneous
("LVD") source at ANY local image tree — it is iterated label-free, since distillation ignores
targets. For true LVD-scale runs supply a WebDataset/tar adapter instead; this folder reader is the
practical substitute and resolves the shipped ``<LVD1689M:root=/PATH>`` config slot.
"""
import logging
import os
from typing import Any, Callable, List, Optional

from .extended import ExtendedVisionDataset

logger = logging.getLogger("eupe")

_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm")


class ImageFolder(ExtendedVisionDataset):
    """Recursively index every image file under ``root`` and serve it with a dummy (0) target.

    Args:
        root: directory tree to scan for images (recursively, following symlinks).
        extra/split: ignored (accepted so the ``name:root=...:extra=...:split=...`` dataset-string
            grammar in eupe.data.loaders parses uniformly with the other datasets).
    """

    def __init__(
        self,
        *,
        root: str,
        extra: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        del extra, split  # label-free folder reader: no per-sample metadata / splits
        self._entries: List[str] = self._index(root)
        if not self._entries:
            raise RuntimeError(f"ImageFolder: no images found under {root!r} (extensions: {_IMG_EXTENSIONS})")
        logger.info("ImageFolder: indexed %d images under %s", len(self._entries), root)

    @staticmethod
    def _index(root: str) -> List[str]:
        paths: List[str] = []
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
            for fn in filenames:
                if fn.lower().endswith(_IMG_EXTENSIONS):
                    paths.append(os.path.join(dirpath, fn))
        paths.sort()  # deterministic order so the rank-aware shuffled samplers are reproducible
        return paths

    def get_image_data(self, index: int) -> bytes:
        with open(self._entries[index], "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return 0  # label-free distillation source; targets are unused

    def __len__(self) -> int:
        return len(self._entries)
