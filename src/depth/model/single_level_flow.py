from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from depth.images.separable_convolution import conv_output_size
from depth.model.patch_flow import PatchFlowEstimator
from depth.patches.extract import extract_patches_nhwc
from depth.patches.extract_shifted import (batch_extract_shifted_patches_nchw,
                                           batch_flow_lands_within_frame)


class LevelFlowEstimator(nnx.Module):
    def __init__(self, stride: int, flow_estimator: PatchFlowEstimator):
        self._flow_estimator = flow_estimator
        self.patch_size = flow_estimator.patch_size
        self.stride = stride

    def __call__(self, frame1: jax.Array, frame2: jax.Array, prior: jax.Array) -> (
            tuple[jax.Array, dict]
    ):
        # shape of img1, img2: (B, H, W, C)
        # shape of prior: (B, PY, PX, 2)
        # returns (B, PY, PX, 3) (dy, dx, match_score)
        assert frame1.shape == frame2.shape
        B, H, W, C = frame1.shape
        PB, PY, PX, F = prior.shape
        assert F == 2
        assert B == PB

        patches1 = extract_patches_nhwc(
            frame1, self.patch_size, self.stride
        )  # B, PY, PX, PH, PW, C

        int_priors = jnp.round(prior).astype(jnp.int32)
        remainder_priors = prior - int_priors

        patches2 = batch_extract_shifted_patches_nchw(frame2, int_priors, self.patch_size,
                                                      self.stride)
        valid_patches = batch_flow_lands_within_frame(int_priors, H, W, self.patch_size,
                                                      self.stride)
        remainder_priors_flat = remainder_priors.reshape(B * PY * PX, 2)
        residual_flow_flat = self._flow_estimator(patches1, patches2, remainder_priors_flat)
        residual_flow, scores = jnp.split(residual_flow_flat.reshape(B, PY, PX, 4), (2,), axis=-1)
        remainder_flow = remainder_priors + residual_flow
        flow = int_priors + remainder_flow
        aux = dict(
            frame1=frame1,
            frame2=frame2,
            valid_patches=valid_patches,
            patches1=patches1,
            patches2=patches2
        )
        assert scores.shape == (B, PY, PX, 2)
        flow_with_scores = jnp.concatenate([flow, scores], axis=-1)
        assert flow_with_scores.shape == (B, PY, PX, 4)
        return (flow_with_scores,  # B, PY, PX, 4 (dy, dx, match_score, patch_score)
                aux)
        # TODO: add patch_score to aux


def test_single_level_flow_estimator():
    rngs = nnx.Rngs(0)
    img = jax.random.uniform(jax.random.key(1), (3, 6, 8, 2))
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=2, train=False, rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)

    patches_y = conv_output_size(6, 4, 2)
    patches_x = conv_output_size(8, 4, 2)
    prior = jax.random.uniform(jax.random.key(2), (3, patches_y, patches_x, 2))
    flow, aux = level_flow_estimator(img, img, prior)
    B, PY, PX, F = prior.shape
    assert flow.shape == (B, PY, PX, 4)
    assert aux['valid_patches'].shape == (B, PY, PX)
    assert aux['valid_patches'].dtype == jnp.bool
    assert aux['patches1'].shape == aux['patches2'].shape == (B, PY, PX, 4, 4, 2)
