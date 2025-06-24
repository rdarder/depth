from dataclasses import dataclass

from flax import nnx as nnx
from jax import numpy as jnp

from depth.images.separable_convolution import conv_output_size_steps, conv_output_size
from depth.model.frame_pair_flow import FramePairFlow
from depth.model.multi_level_flow import PyramidFlowEstimator
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.pyramid import ImageDownscaler, ImagePyramidDecomposer
from depth.model.single_level_flow import LevelFlowEstimator


@dataclass
class ModelSettings:
    img_size: int = 158
    levels: int = 4
    decompose_kernel_size: int = 4
    decompose_stride: int = 2
    patch_size: int = 4
    patch_stride: int = 2


def make_model(seed: int, train: bool, settings: ModelSettings) \
        -> (
                FramePairFlow):
    rngs = nnx.Rngs(seed)
    image_downscaler = ImageDownscaler(alpha=0.5, stride=settings.decompose_stride)
    pyramid_decomposer = ImagePyramidDecomposer(image_downscaler, levels=settings.levels)
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=settings.patch_size, num_channels=1, features_dim=8, train=train, rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=settings.patch_stride,
                                              flow_estimator=patch_flow_estimator)
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator)
    frame_flow_estimator = FramePairFlow(pyramid_flow_estimator, pyramid_decomposer)
    return frame_flow_estimator


def generate_zero_priors(batch_size, settings: ModelSettings):
    # pyramid decomposition
    decomposed_patches = conv_output_size_steps(settings.img_size, settings.decompose_kernel_size,
                                                settings.decompose_stride, settings.levels)
    extracted_patches = conv_output_size(decomposed_patches, settings.patch_size,
                                         settings.patch_stride)
    initial_priors = jnp.zeros((batch_size, extracted_patches, extracted_patches, 2),
                               dtype=jnp.float32)
    return initial_priors
