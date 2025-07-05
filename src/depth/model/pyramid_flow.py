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
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimator, LevelFlowEstimationParams, \
    LevelFlowEstimation
from depth.model.upsample import FlowUpsampler


@register_dataclass
@dataclass
class PyramidFlowEstimationParams:
    pyramid1: Sequence[jax.Array]
    pyramid2: Sequence[jax.Array]
    prior: jax.Array
    confidence: jax.Array

    def check_shapes_consistent(self, patch_size: int, stride: int):
        assert all(f1.shape == f2.shape for f1, f2 in zip(self.pyramid1, self.pyramid2))
        coarsest_grained_frame = self.pyramid1[-1]
        B, H, W, C = coarsest_grained_frame.shape
        LH = conv_output_size(H, patch_size, stride)
        LW = conv_output_size(W, patch_size, stride)

        expected_prior_shape = (
            B,
            LH, LW,
            2
        )
        assert self.prior.shape == expected_prior_shape
        assert self.confidence.shape == (B, LH, LW, 1)


@register_dataclass
@dataclass
class PyramidFlowEstimation:
    pyramid: Sequence[LevelFlowEstimation]

    def check_shapes_consistent(self, patch_size: int, stride: int):
        B, H, W, F = self.pyramid[0].net_flow.shape
        for i, level_estimation in enumerate(self.pyramid[:-1]):
            level_estimation.check_shapes_consistent()
            assert level_estimation.net_flow.shape == (B, H, W, F)
            if i < len(self.pyramid) - 1:
                H = conv_output_size(H, patch_size, stride)
                W = conv_output_size(W, patch_size, stride)


class PyramidFlowEstimator(nnx.Module):
    def __init__(self, level_flow_estimator: LevelFlowEstimator):
        self._level_flow_estimator = level_flow_estimator

    def __call__(self, params: PyramidFlowEstimationParams) -> PyramidFlowEstimation:
        params.check_shapes_consistent(self._level_flow_estimator.patch_size,
                                       self._level_flow_estimator.stride)
        estimation_pyramid = []
        confidence = params.confidence
        prior = params.prior
        for frame1, frame2 in zip(reversed(params.pyramid1), reversed(params.pyramid2)):
            level_params = LevelFlowEstimationParams(
                frame1=frame1,
                frame2=frame2,
                prior=prior,
                confidence=confidence,
            )
            level_estimation = self._level_flow_estimator(level_params)
            level_estimation.check_shapes_consistent()
            estimation_pyramid.append(level_estimation)
            prior = level_estimation.upsampled_flow.upsampled_flow
            confidence = level_estimation.upsampled_flow.fwd_confidence
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
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator,
                                              upsampler=upsampler)
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator)
    pyramid1 = build_image_pyramid(batch1, levels=5, keep=5)
    pyramid2 = build_image_pyramid(batch2, levels=5, keep=5)
    prior = jnp.zeros((2, 3, 3, 2), jnp.float32)
    confidence = jnp.zeros((2, 3, 3, 1), jnp.float32)
    pyramid_flow_params = PyramidFlowEstimationParams(
        pyramid1=pyramid1,
        pyramid2=pyramid2,
        prior=prior,
        confidence=confidence
    )
    flow: PyramidFlowEstimation = pyramid_flow_estimator(pyramid_flow_params)
    jax.block_until_ready(flow)
    flow.check_shapes_consistent(patch_size=4, stride=2)
