from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainSettings:
    learning_rate: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 100
    max_frames_lookahead: int = 8
    tensorboard_logs: Path = Path('./runs')
    checkpoint_dir: Path = Path('./checkpoints')
    train_dataset_root: Path = Path('./datasets/frames')


@dataclass
class ModelSettings:
    img_size: int = 190
    levels: int = 6
    decompose_kernel_size: int = 4
    decompose_stride: int = 2
    patch_size: int = 4
    patch_stride: int = 2
    mlp_hidden_size: int = 16
    predictor_features: int = 8


@dataclass
class Settings:
    model: ModelSettings
    train: TrainSettings
