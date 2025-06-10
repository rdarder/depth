from functools import lru_cache
from itertools import chain, batched
from pathlib import Path
from random import Random

from typing import Iterable, Iterator

import cv2
from dataclasses import dataclass
import jax.numpy as jnp
import jax


@dataclass(frozen=True)
class FrameMetadata:
    video: str
    filename: str


def load_frame_from_path(frame_path: str):
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f'frame not found at {frame_path}')
    as_array = jnp.asarray(img)
    assert as_array.ndim == 2
    normalized = as_array / 255.0
    with_channel_dimension = normalized[None, :, :]
    return with_channel_dimension


class FrameSource:
    def __init__(self, folders: Iterable[Path], cache_size: int = 0):
        self.frames = self.discover_video_frames(sorted(folders))
        if cache_size > 0:
            self.loader = lru_cache(maxsize=cache_size)(load_frame_from_path)
        else:
            self.loader = load_frame_from_path

    def discover_video_frames(self, folders: Iterable[Path]):
        global_index = 0
        all_frame_metadata = []
        for folder in folders:
            frame_paths = sorted(folder.glob("*.png"))
            for local_idx, path in enumerate(frame_paths, start=1):
                frame = FrameMetadata(video=str(path.parent), filename=path.name)
                all_frame_metadata.append(frame)
                global_index += 1
        return all_frame_metadata

    def make_frame_pairs(self, max_frame_lookahead=1) -> Iterable[tuple[int, int]]:
        for i in range(len(self.frames)):
            for j in range(max_frame_lookahead):
                target = i + j + 1
                if target < len(self.frames) and self.frames[target].video == self.frames[i].video:
                    yield (i, target)

    def add_reversed_pairs(self, frame_pairs: Iterable[tuple[int, int]]) -> (
            Iterable[tuple[int, int]]):
        for pair in frame_pairs:
            yield pair
            yield (pair[1], pair[0])

    def load_single_frame(self, frame_idx: int) -> jax.Array:
        frame_metadata = self.frames[frame_idx]
        frame_path = f"{frame_metadata.video}/{frame_metadata.filename}"
        return self.loader(frame_path)

    def load_frame_pair(self, pair: tuple[int, int]):
        return self.load_single_frame(pair[0]), self.load_single_frame(pair[1])


class FramePairsDataset:
    def __init__(self, source: FrameSource, batch_size: int = 10, max_frame_lookahead=1,
                 include_reversed_pairs=True, drop_uneven_batches=False, seed: int = 0):
        self._source = source
        self._batch_size = batch_size
        self._max_frame_lookahead = max_frame_lookahead
        self._include_reversed_pairs = include_reversed_pairs
        self._random = Random(seed)
        self._drop_uneven_batches = drop_uneven_batches
        self._frame_pairs = list(self._make_frame_pairs())

    def _make_frame_pairs(self):
        frame_pairs = self._source.make_frame_pairs(self._max_frame_lookahead)
        if self._include_reversed_pairs:
            frame_pairs = self._source.add_reversed_pairs(frame_pairs)
        return frame_pairs

    def __len__(self):
        return len(self._frame_pairs)

    def __iter__(self) -> Iterator[jax.Array]:
        shuffled_pairs = self._random.sample(self._frame_pairs, len(self._frame_pairs))
        for batch in batched(shuffled_pairs, self._batch_size):
            if self._drop_uneven_batches and len(batch) < self._batch_size:
                break
            loaded_pairs = [self._source.load_frame_pair(pair) for pair in batch]
            batch_array = jnp.array(loaded_pairs).transpose(1, 0, 2, 3, 4)
            yield batch_array

def sample_run():
    source = FrameSource(Path('datasets/frames').glob('*'), cache_size=10_000)
    dataset = FramePairsDataset(
        source,
        batch_size=100,
        max_frame_lookahead=5,
        include_reversed_pairs=True,
        drop_uneven_batches=True,
        seed=0
    )
    print(f'dataset length: {len(dataset)}')
    for i in range(10):
        print(f"epoch {i}")
        for j, f in enumerate(dataset):
            if i == 0 and j == 0:
                print(f"frame batch shape: {f.shape}")
            if j % 100 == 0:
                print(f"batch {j}")

if __name__ == '__main__':
    sample_run()