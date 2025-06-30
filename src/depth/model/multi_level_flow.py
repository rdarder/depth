from importlib import resources
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from depth.images.load import load_frame_from_path
from depth.images.pyramid import build_image_pyramid
from depth.images.separable_convolution import conv_output_size
from depth.images.upscale import upscale_size_2n_plus_2, upscale_values_2n_plus2
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimator


class PyramidFlowEstimator(nnx.Module):
    def __init__(self, level_flow_estimator: LevelFlowEstimator):
        self._level_flow_estimator = level_flow_estimator
        self.patch_size = level_flow_estimator.patch_size
        self.stride = level_flow_estimator.stride

    def _check_prior_shape(self, coarsest_grained_frame: jax.Array, prior: jax.Array):
        B, H, W, C = coarsest_grained_frame.shape
        expected_prior_shape = (
            B,
            conv_output_size(H, self.patch_size, self.stride),
            conv_output_size(W, self.patch_size, self.stride),
            2
        )
        assert prior.shape == expected_prior_shape

    def __call__(self, pyramid1: Sequence[jax.Array], pyramid2: Sequence[jax.Array],
                 prior: jax.Array) -> tuple[Sequence[jax.Array], Sequence[dict]]:
        flow_pyramid = []
        aux_pyramid = []
        self._check_prior_shape(pyramid1[-1], prior)
        for img1, img2 in zip(reversed(pyramid1), reversed(pyramid2)):
            flow, aux = self._level_flow_estimator(img1, img2, prior)
            flow_pyramid.append(flow)
            aux_pyramid.append(aux)
            upscaled_values = upscale_values_2n_plus2(flow)
            prior = upscale_size_2n_plus_2(upscaled_values)
        return flow_pyramid[::-1], aux_pyramid[::-1]


def test_multi_level_flow_estimator():
    frame1_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame2_path = resources.files('depth.test_fixtures') / "frame2.png"
    frame1 = load_frame_from_path(str(frame1_path), 158)
    frame2 = load_frame_from_path(str(frame2_path), 158)
    batch1 = jnp.stack([frame1, frame2], axis=0)
    batch2 = jnp.stack([frame2, frame1], axis=0)
    rngs = nnx.Rngs(0)
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=1, train=False, rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator)
    pyramid1 = build_image_pyramid(batch1, levels=5, keep=5)
    pyramid2 = build_image_pyramid(batch2, levels=5, keep=5)
    prior = jnp.zeros((2, 3, 3, 2), jnp.float32)
    flow_pyramid, aux_pyramid = pyramid_flow_estimator(pyramid1, pyramid2, prior)
    jax.block_until_ready(flow_pyramid)
    assert flow_pyramid[-1].shape == (2, 3, 3, 2)
    assert flow_pyramid[-2].shape == (2, 8, 8, 2)
    assert flow_pyramid[-3].shape == (2, 18, 18, 2)
    assert flow_pyramid[-4].shape == (2, 38, 38, 2)
    assert flow_pyramid[-5].shape == (2, 78, 78, 2)
    assert aux_pyramid[-1]['patches1'].shape == (2, 3, 3, 4, 4, 1)
    assert aux_pyramid[-1]['patches2'].shape == (2, 3, 3, 4, 4, 1)
    assert aux_pyramid[-1]['valid_patches'].shape == (2, 3, 3)
