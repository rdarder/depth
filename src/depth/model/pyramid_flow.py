from importlib import resources
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from depth.images.load import load_frame_from_path
from depth.images.pyramid import build_image_pyramid
from depth.images.separable_convolution import conv_output_size
from depth.images.upscale import upscale_values_2n_plus2
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimator
from depth.model.upscale import FlowUpscaler


class PyramidFlowEstimator(nnx.Module):
    def __init__(self, level_flow_estimator: LevelFlowEstimator, upscaler: FlowUpscaler):
        self._level_flow_estimator = level_flow_estimator
        self._upscaler = upscaler
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
        B, H, W, F = prior.shape
        confidence = jnp.ones((B, H, W, 1), jnp.float32) * 0.5  # should probably come as a param.
        self._check_prior_shape(pyramid1[-1], prior)
        for img1, img2 in zip(reversed(pyramid1), reversed(pyramid2)):
            flow_with_scores, aux = self._level_flow_estimator(img1, img2, prior)
            flow_pyramid.append(flow_with_scores)
            aux['confidence'] = confidence
            aux_pyramid.append(aux)
            upscale_input = jnp.concatenate([flow_with_scores, confidence], axis=-1)
            upscaled_flow, confidence = self._upscaler(upscale_input)
            prior = upscale_values_2n_plus2(upscaled_flow)
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
    upscaler = FlowUpscaler(rngs=rngs)
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator, upscaler=upscaler)
    pyramid1 = build_image_pyramid(batch1, levels=5, keep=5)
    pyramid2 = build_image_pyramid(batch2, levels=5, keep=5)
    prior = jnp.zeros((2, 3, 3, 2), jnp.float32)
    flow_pyramid, aux_pyramid = pyramid_flow_estimator(pyramid1, pyramid2, prior)
    jax.block_until_ready(flow_pyramid)
    assert flow_pyramid[-1].shape == (2, 3, 3, 4)
    assert flow_pyramid[-2].shape == (2, 8, 8, 4)
    assert flow_pyramid[-3].shape == (2, 18, 18, 4)
    assert flow_pyramid[-4].shape == (2, 38, 38, 4)
    assert flow_pyramid[-5].shape == (2, 78, 78, 4)
    assert aux_pyramid[-1]['patches1'].shape == (2, 3, 3, 4, 4, 1)
    assert aux_pyramid[-1]['patches2'].shape == (2, 3, 3, 4, 4, 1)
    assert aux_pyramid[-1]['valid_patches'].shape == (2, 3, 3)
