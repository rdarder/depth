from depth.datasets.frames import FramePairsDataset, FrameSource
from depth.datasets.pyramids import FramePyramidPairsDataset
from depth.model.settings import TrainSettings, Settings


def make_frame_pyramids_dataset(settings: Settings, levels: int):
    source = FrameSource(settings.train.train_dataset_root.glob('*'),
                         img_size=settings.model.img_size)

    frame_pairs = FramePairsDataset(
        source,
        batch_size=settings.train.batch_size,
        max_frame_lookahead=settings.train.max_frames_lookahead,
        include_reversed_pairs=True,
        drop_uneven_batches=True,
        seed=1
    )
    pyramid_pairs = FramePyramidPairsDataset(frame_pairs, settings.model.levels, keep=levels,
                                             cache_size=10_000)
    return pyramid_pairs
