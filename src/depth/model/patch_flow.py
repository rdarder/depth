from __future__ import annotations

import jax
from flax import nnx
from flax.nnx import Rngs
from jax import numpy as jnp

from depth.images.separable_convolution import conv_output_size
from depth.patches.extract import extract_patches_nhwc


class PatchFlowEstimator(nnx.Module):
    def __init__(self, patch_size: int, num_channels: int, features: int = 32,
                 mlp_hidden_size: int = 16, *, train: bool, rngs: Rngs):
        assert patch_size > 3
        self.patch_size = patch_size

        self.shift_conv = nnx.Conv(
            in_features=2 * num_channels,
            out_features=8 * num_channels,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding='VALID',
            feature_group_count=2 * num_channels,
            rngs=rngs,
            use_bias=False,
        )
        self.mix_shifts_conv = nnx.Conv(
            in_features=8 * num_channels,
            out_features=30,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='VALID',
            rngs=rngs,
            use_bias=False,
        )
        self.bn_mix_shifts = nnx.BatchNorm(num_features=30, use_running_average=not train,
                                           rngs=rngs)
        self.mlp_hidden = nnx.Linear(
            in_features=32,
            out_features=16,
            use_bias=True,
            rngs=rngs,
        )
        self.bn_hidden1 = nnx.BatchNorm(num_features=16, use_running_average=not train, rngs=rngs)
        self.mlp_hidden2 = nnx.Linear(
            in_features=16,
            out_features=16,
            use_bias=True,
            rngs=rngs,
        )
        self.bn_hidden2 = nnx.BatchNorm(num_features=16, use_running_average=not train, rngs=rngs)
        self.mlp_output = nnx.Linear(
            in_features=16,
            out_features=2,  # TODO: add confidence prediction here.
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, patch1: jnp.ndarray, patch2: jnp.ndarray, prior: jnp.ndarray):
        B, PH, PW, H, W, C = patch1.shape
        BP = B * PH * PW
        patches = jnp.stack([patch1, patch2], axis=-1).reshape(BP, H, W, C * 2)
        flat_priors = prior.reshape(B * PH * PW, 2)
        shifted_patches = self.shift_conv(patches)
        mixed_shifts = self.mix_shifts_conv(shifted_patches)
        bn_mixed_shifts = self.bn_mix_shifts(mixed_shifts)
        avg_abs_mixed_shifts = jnp.mean(jnp.abs(bn_mixed_shifts), axis=(1, 2)).reshape(BP, -1)
        avg_shifts_and_priors = jnp.concatenate([avg_abs_mixed_shifts, flat_priors], axis=-1)
        hidden_state = self.mlp_hidden(avg_shifts_and_priors)
        bn_hidden_state = self.bn_hidden1(hidden_state)
        non_linear_hidden = jax.nn.relu(bn_hidden_state)
        hidden_state2 = self.mlp_hidden2(non_linear_hidden)
        bn_hidden_state2 = self.bn_hidden2(hidden_state2)
        non_linear_hidden2 = jax.nn.relu(bn_hidden_state2)
        output = self.mlp_output(non_linear_hidden2)
        norm_output = jax.nn.tanh(output)
        norm_output_grid = norm_output.reshape(B, PH, PW, 2)  # add confidence here.
        return norm_output_grid


def test_patch_flow_estimator():
    rngs = nnx.Rngs(0)
    estimator = PatchFlowEstimator(patch_size=4, num_channels=2, features=8, mlp_hidden_size=16,
                                   train=False, rngs=rngs)
    canvas = jax.random.uniform(jax.random.key(0), (2, 7, 9, 2))  # B,H,W,C
    frame1 = canvas[:, 1:, 1:, :]
    frame2 = canvas[:, :-1, :-1, :]
    patches1 = extract_patches_nhwc(frame1, patch_size=4, stride=2)
    patches2 = extract_patches_nhwc(frame2, patch_size=4, stride=2)
    priors = jnp.zeros((2, 2, 3, 2))
    # Pass use_running_average to the __call__ method
    flow_delta = estimator(patches1, patches2, priors)
    assert flow_delta.shape == priors.shape


def test_patch_flow_estimator_patch_size5():
    rngs = nnx.Rngs(0)
    estimator = PatchFlowEstimator(patch_size=5, num_channels=2, features=8, mlp_hidden_size=16,
                                   train=False, rngs=rngs)
    canvas = jax.random.uniform(jax.random.key(0), (2, 8, 10, 2))  # B,H,W,C
    frame1 = canvas[:, 1:, 1:, :]
    frame2 = canvas[:, :-1, :-1, :]
    patches1 = extract_patches_nhwc(frame1, patch_size=5, stride=2)
    patches2 = extract_patches_nhwc(frame2, patch_size=5, stride=2)
    priors = jnp.zeros((2, 2, 3, 2))
    # Pass use_running_average to the __call__ method
    flow_delta = estimator(patches1, patches2, priors)
    assert flow_delta.shape == priors.shape
