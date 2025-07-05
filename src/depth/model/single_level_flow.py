from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import register_dataclass

from depth.images.separable_convolution import conv_output_size
from depth.model.patch_flow import PatchFlowEstimator, PatchFlowEstimationParams, \
    PatchFlowEstimation
from depth.model.upsample import FlowUpsampler, UpsampledFlow, FlowUpsamplerParams
from depth.patches.extract import extract_patches_nhwc
from depth.patches.extract_shifted import (batch_extract_shifted_patches_nchw,
                                           batch_flow_lands_within_frame)


@register_dataclass
@dataclass
class LevelFlowEstimationParams:
    frame1: jax.Array
    frame2: jax.Array
    prior: jax.Array
    confidence: jax.Array

    def check_shapes_consistent(self, patch_size: int, stride: int):
        assert self.frame1.shape == self.frame2.shape
        B, H, W, C = self.frame1.shape
        PB, PY, PX, F = self.prior.shape
        assert B == PB
        assert F == 2
        assert PY == conv_output_size(H, patch_size, stride)
        assert PX == conv_output_size(W, patch_size, stride)


@register_dataclass
@dataclass
class LevelFlowEstimation:
    frame1: jax.Array
    frame2: jax.Array
    valid_patches: jax.Array
    patches1: jax.Array
    patches2: jax.Array
    match_score: jax.Array
    patch_score: jax.Array
    residual_flow: jax.Array
    net_flow: jax.Array
    upsampled_flow: UpsampledFlow

    def check_shapes_consistent(self):
        assert self.frame1.shape == self.frame2.shape
        B, PY, PX, F = self.net_flow.shape
        assert self.valid_patches.shape == (B, PY, PX)
        assert self.valid_patches.dtype == jnp.bool
        assert self.valid_patches.shape == (B, PY, PX)
        assert self.patches2.shape == self.patches1.shape
        assert self.patches1.shape[:3] == (B, PY, PX)
        assert self.match_score.shape == (B, PY, PX, 1)
        assert self.patch_score.shape == (B, PY, PX, 1)
        assert self.residual_flow.shape == (B, PY, PX, 2)
        assert self.net_flow.shape == (B, PY, PX, 2)
        self.upsampled_flow.check_shapes_consistent()

    def check_shapes_consistent_with_params(self, params: LevelFlowEstimationParams, patch_size:
    int, stride: int):
        self.check_shapes_consistent()
        params.check_shapes_consistent(patch_size, stride)
        assert self.frame1.shape == params.frame1.shape
        B, H, W, C = self.frame1.shape
        PY = conv_output_size(H, patch_size, stride)
        PX = conv_output_size(W, patch_size, stride)
        assert params.prior.shape == (B, PY, PX, 2)
        assert self.valid_patches.shape == (B, PY, PX)
        assert self.patches1.shape == (B, PY, PX, patch_size, patch_size, C)
        assert self.patches2.shape == self.patches1.shape
        assert self.match_score.shape == (B, PY, PX, 1)
        assert self.patch_score.shape == (B, PY, PX, 1)
        assert self.residual_flow.shape == (B, PY, PX, 2)
        assert self.net_flow.shape == (B, PY, PX, 2)


class LevelFlowEstimator(nnx.Module):
    def __init__(self, stride: int, flow_estimator: PatchFlowEstimator, upsampler: FlowUpsampler):
        self._flow_estimator = flow_estimator
        self._upsampler = upsampler
        self.patch_size = flow_estimator.patch_size
        self.stride = stride

    def __call__(self, params: LevelFlowEstimationParams) -> LevelFlowEstimation:
        params.check_shapes_consistent(self.patch_size, self.stride)
        B, H, W, C = params.frame1.shape
        PB, PY, PX, F = params.prior.shape

        patches1 = extract_patches_nhwc(
            params.frame1, self.patch_size, self.stride
        )  # B, PY, PX, PH, PW, C

        int_priors = jnp.round(params.prior).astype(jnp.int32)
        remainder_priors = params.prior - int_priors

        patches2 = batch_extract_shifted_patches_nchw(
            params.frame2, int_priors, self.patch_size, self.stride
        )
        valid_patches = batch_flow_lands_within_frame(
            int_priors, H, W, self.patch_size, self.stride
        )
        # remainder_priors_flat = remainder_priors.reshape(B * PY * PX, 2)

        patch_params = PatchFlowEstimationParams(
            patch1=patches1,
            patch2=patches2,
            prior=remainder_priors
        )

        estimation: PatchFlowEstimation = self._flow_estimator(patch_params)
        net_flow = params.prior + estimation.residual_flow
        upsample_params = FlowUpsamplerParams(
            residual_flow=estimation.residual_flow,
            net_flow=net_flow,
            confidence=params.confidence,
            match_score=estimation.match_score,
            patch_score=estimation.patch_score,
        )
        upsampled: UpsampledFlow = self._upsampler(upsample_params)

        level_estimation = LevelFlowEstimation(
            frame1=params.frame1,
            frame2=params.frame2,
            valid_patches=valid_patches,
            patches1=patches1,
            patches2=patches2,
            match_score=estimation.match_score,
            patch_score=estimation.patch_score,
            residual_flow=estimation.residual_flow,
            net_flow=params.prior + estimation.residual_flow,
            upsampled_flow=upsampled
        )
        return level_estimation


def test_single_level_flow_estimator():
    rngs = nnx.Rngs(0)
    img = jax.random.uniform(jax.random.key(1), (3, 6, 8, 2))
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=2, train=False, rngs=rngs
    )
    upsampler = FlowUpsampler(rngs=rngs)
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator,
                                              upsampler=upsampler)

    patches_y = conv_output_size(6, 4, 2)
    patches_x = conv_output_size(8, 4, 2)
    prior = jax.random.uniform(jax.random.key(2), (3, patches_y, patches_x, 2))
    confidence = jax.random.uniform(jax.random.key(3), (3, patches_y, patches_x, 1))

    params = LevelFlowEstimationParams(
        frame1=img,
        frame2=img,
        prior=prior,
        confidence=confidence,
    )
    estimation: LevelFlowEstimation = level_flow_estimator(params)
    estimation.check_shapes_consistent_with_params(params, patch_size=4, stride=2)
