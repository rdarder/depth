from functools import lru_cache
from typing import Sequence

import jax

from depth.datasets.frames import FramePairsDataset
from depth.images.pyramid import build_image_pyramid


class FramePyramidPairsDataset:
    """An adapter for FramePairsDataset that returns and caches pyramids at truncated levels."""

    def __init__(self, frame_pairs_dataset: FramePairsDataset, levels: int, keep: int,
                 cache_size: int = 0):
        self._frame_pairs_dataset = frame_pairs_dataset
        self._levels = levels
        self._keep = keep
        if cache_size > 0:
            self.loader = lru_cache(maxsize=cache_size)(self._frame_to_pyramid)
        else:
            self.loader = self._frame_to_pyramid

    def __len__(self):
        return len(self._frame_pairs_dataset)

    def __iter__(self):
        for f1, f2 in self._frame_pairs_dataset:
            yield self._frame_to_pyramid(f1), self._frame_to_pyramid(f2)

    def _frame_to_pyramid(self, frame: jax.Array) -> Sequence[jax.Array]:
        return build_image_pyramid(frame, self._levels, self._keep)
