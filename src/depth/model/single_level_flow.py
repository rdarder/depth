from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import nnx

from depth.images.separable_convolution import conv_output_size
from depth.model.patch_flow import PatchFlowEstimator
from depth.patches.extract import extract_patches_nhwc
from depth.patches.extract_shifted import batch_extract_shifted_patches_nchw
from depth.patches.loss_grid import patch_flow_loss_grid


class LevelFlowEstimator(nnx.Module):
    def __init__(self, stride: int, flow_estimator: PatchFlowEstimator):
        self._flow_estimator = flow_estimator
        self.patch_size = flow_estimator.patch_size
        self.stride = stride

    def __call__(self, img1: jax.Array, img2: jax.Array, prior: jax.Array):
        # shape of img1, img2: (B, H, W, C)
        # shape of prior: (B, PY, PX, 2)
        # returns (B, PY, PX, 3) (dy, dx, loss)
        assert img1.shape == img2.shape
        B, PY, PX, F = prior.shape
        assert F == 2

        patches1 = extract_patches_nhwc(
            img1, self.patch_size, self.stride
        )  # B, PY, PX, PH, PW, C

        int_priors = jnp.round(prior).astype(jnp.int32)
        remainder_priors = prior - int_priors

        patches2, patch_validity = batch_extract_shifted_patches_nchw(
            img2, int_priors, self.patch_size, self.stride
        )
        remainder_priors_flat = remainder_priors.reshape(B * PY * PX, 2)
        residual_flow_flat = self._flow_estimator(patches1, patches2, remainder_priors_flat)
        residual_flow = residual_flow_flat.reshape(B, PY, PX, 2)
        remainder_flow = remainder_priors + residual_flow
        flow = int_priors + remainder_flow
        loss = patch_flow_loss_grid(patches1, patches2, remainder_flow)[:, :, :, None]
        flow_with_loss = jnp.concatenate(
            [flow, loss], axis=-1
        )  # B, PY, PX, 3 (dy, dx, loss)
        return flow_with_loss


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
    flow = level_flow_estimator(img, img, prior)
    B, PY, PX, F = prior.shape
    assert flow.shape == (B, PY, PX, F + 1)
