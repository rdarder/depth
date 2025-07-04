from dataclasses import dataclass
from importlib import resources
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import register_dataclass

from depth.images.load import load_frame_from_path
from depth.images.pyramid import build_image_pyramid
from depth.images.separable_convolution import conv_output_size
from depth.images.upscale import upsample_2n_plus2, upscale_values_2n_plus2
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimator, LevelFlowEstimationParams, \
    LevelFlowEstimation
from depth.model.upsample import FlowUpsampler, FlowUpsamplerParams


@register_dataclass
@dataclass
class PyramidFlowEstimationParams:
    pyramid1: Sequence[jax.Array]
    pyramid2: Sequence[jax.Array]
    prior: jax.Array


@register_dataclass
@dataclass
class PyramidFlowEstimation:
    pyramid: Sequence[LevelFlowEstimation]


class PyramidFlowEstimator(nnx.Module):
    def __init__(self, level_flow_estimator: LevelFlowEstimator, upsampler: FlowUpsampler):
        self._level_flow_estimator = level_flow_estimator
        self._upsampler = upsampler
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

    def __call__(self, params: PyramidFlowEstimationParams) -> PyramidFlowEstimation:
        estimation_pyramid = []
        B, H, W, F = params.prior.shape
        confidence = jnp.ones((B, H, W, 1), jnp.float32) * 0.5  # should probably come as a param.
        self._check_prior_shape(params.pyramid1[-1], params.prior)
        prior = params.prior
        for frame1, frame2 in zip(reversed(params.pyramid1), reversed(params.pyramid2)):
            level_params = LevelFlowEstimationParams(
                frame1=frame1,
                frame2=frame2,
                prior=prior
            )
            level_estimation = self._level_flow_estimator(level_params)
            estimation_pyramid.append(level_estimation)
            upsample_params = FlowUpsamplerParams(
                residual_flow=level_estimation.residual_flow,
                net_flow=level_estimation.net_flow,
                confidence=confidence,
                match_score=level_estimation.match_score,
                patch_score=level_estimation.patch_score,
            )
            upsampled = self._upsampler(upsample_params)
            prior = upsampled.upsampled_flow
            confidence = upsampled.fwd_confidence
        reversed_pyramid = estimation_pyramid[::-1]
        return PyramidFlowEstimation(pyramid=reversed_pyramid)


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
    upsampler = FlowUpsampler(rngs=rngs)
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator, upsampler=upsampler)
    pyramid1 = build_image_pyramid(batch1, levels=5, keep=5)
    pyramid2 = build_image_pyramid(batch2, levels=5, keep=5)
    prior = jnp.zeros((2, 3, 3, 2), jnp.float32)
    pyramid_flow_params = PyramidFlowEstimationParams(
        pyramid1=pyramid1,
        pyramid2=pyramid2,
        prior=prior,
    )
    flow: PyramidFlowEstimation = pyramid_flow_estimator(pyramid_flow_params)
    jax.block_until_ready(flow)

    assert flow.pyramid[-1].net_flow.shape == (2, 3, 3, 2)
    assert flow.pyramid[-2].net_flow.shape == (2, 8, 8, 2)
    assert flow.pyramid[-3].net_flow.shape == (2, 18, 18, 2)
    assert flow.pyramid[-4].net_flow.shape == (2, 38, 38, 2)
    assert flow.pyramid[-5].net_flow.shape == (2, 78, 78, 2)
    assert flow.pyramid[-1].patches1.shape == (2, 3, 3, 4, 4, 1)
    assert flow.pyramid[-1].patches2.shape == (2, 3, 3, 4, 4, 1)
    assert flow.pyramid[-1].valid_patches.shape == (2, 3, 3)
