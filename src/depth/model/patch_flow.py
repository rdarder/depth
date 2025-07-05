from __future__ import annotations

from dataclasses import dataclass

import jax
from flax import nnx
from flax.nnx import Rngs
from jax import numpy as jnp
from jax.tree_util import register_dataclass

from depth.patches.extract import extract_patches_nhwc


def patches_score(patches: jax.Array) -> jax.Array:
    patches_std_channel = jnp.std(patches, axis=(3, 4))
    patches_std = jnp.mean(patches_std_channel, axis=-1)[:, :, :, None]
    lam = 50.0
    patches_scores = 1 - (jnp.exp(-lam * patches_std))
    return patches_scores


@register_dataclass
@dataclass
class PatchFlowEstimationParams:
    patch1: jax.Array
    patch2: jax.Array
    prior: jax.Array

    def check_shapes_consistent(self):
        assert self.patch1.shape == self.patch2.shape
        B, PH, PW, H, W, C = self.patch1.shape
        assert self.prior.shape == (B, PH, PW, 2)


@register_dataclass
@dataclass
class PatchFlowEstimation:
    residual_flow: jax.Array
    patch_score: jax.Array
    match_score: jax.Array

    def check_shapes_consistent_with_params(self, params: PatchFlowEstimationParams):
        params.check_shapes_consistent()
        B, PH, PW, F = params.prior.shape
        assert self.residual_flow.shape == (B, PH, PW, 2)
        assert self.match_score.shape == (B, PH, PW, 1)
        assert self.patch_score.shape == (B, PH, PW, 1)


class PatchFlowEstimator(nnx.Module):
    def __init__(self, patch_size: int, num_channels: int, *, train: bool, rngs: Rngs):
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
            out_features=3,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, params: PatchFlowEstimationParams) -> PatchFlowEstimation:
        B, PH, PW, H, W, C = params.patch1.shape
        BP = B * PH * PW
        patches = jnp.stack([params.patch1, params.patch2], axis=-1).reshape(BP, H, W, C * 2)
        flat_priors = params.prior.reshape(B * PH * PW, 2)
        shifted_patches = self.shift_conv(patches)
        mixed_shifts = self.mix_shifts_conv(shifted_patches)
        bn_mixed_shifts = self.bn_mix_shifts(mixed_shifts)
        avg_abs_mixed_shifts = jnp.mean(jnp.abs(bn_mixed_shifts), axis=(1, 2)).reshape(BP, -1)
        avg_shifts_priors_and_std = jnp.concatenate(
            [avg_abs_mixed_shifts, flat_priors],
            axis=-1)
        hidden_state = self.mlp_hidden(avg_shifts_priors_and_std)
        bn_hidden_state = self.bn_hidden1(hidden_state)
        non_linear_hidden = jax.nn.relu(bn_hidden_state)
        hidden_state2 = self.mlp_hidden2(non_linear_hidden)
        bn_hidden_state2 = self.bn_hidden2(hidden_state2)
        non_linear_hidden2 = jax.nn.relu(bn_hidden_state2)
        output = self.mlp_output(non_linear_hidden2)
        norm_output = jax.nn.tanh(output)
        norm_output_grid = norm_output.reshape(B, PH, PW, 3)
        norm_flow_grid, match_score_grid = jnp.split(norm_output_grid, (2,), axis=-1)
        patch2_score = patches_score(params.patch2)
        estimation = PatchFlowEstimation(
            residual_flow=norm_flow_grid,
            patch_score=patch2_score,
            match_score=match_score_grid
        )
        return estimation


def test_patch_flow_estimator():
    rngs = nnx.Rngs(0)
    estimator = PatchFlowEstimator(patch_size=4, num_channels=2, train=False, rngs=rngs)
    canvas = jax.random.uniform(jax.random.key(0), (2, 7, 9, 2))  # B,H,W,C
    frame1 = canvas[:, 1:, 1:, :]
    frame2 = canvas[:, :-1, :-1, :]
    patches1 = extract_patches_nhwc(frame1, patch_size=4, stride=2)
    patches2 = extract_patches_nhwc(frame2, patch_size=4, stride=2)
    priors = jnp.zeros((2, 2, 3, 2))
    params = PatchFlowEstimationParams(
        patch1=patches1,
        patch2=patches2,
        prior=priors
    )
    estimation: PatchFlowEstimation = estimator(params)
    estimation.check_shapes_consistent_with_params(params)


def test_patch_flow_estimator_patch_size5():
    rngs = nnx.Rngs(0)
    estimator = PatchFlowEstimator(patch_size=5, num_channels=2, train=False, rngs=rngs)
    canvas = jax.random.uniform(jax.random.key(0), (2, 8, 10, 2))  # B,H,W,C
    frame1 = canvas[:, 1:, 1:, :]
    frame2 = canvas[:, :-1, :-1, :]
    patches1 = extract_patches_nhwc(frame1, patch_size=5, stride=2)
    patches2 = extract_patches_nhwc(frame2, patch_size=5, stride=2)
    priors = jnp.zeros((2, 2, 3, 2))
    params = PatchFlowEstimationParams(
        patch1=patches1,
        patch2=patches2,
        prior=priors
    )
    estimation: PatchFlowEstimation = estimator(params)
    estimation.check_shapes_consistent_with_params(params)
