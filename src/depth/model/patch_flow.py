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
        total_input_channels = 2 * num_channels + 2
        self.conv1_1x1 = nnx.Conv(in_features=total_input_channels, out_features=features,
                                  kernel_size=(1, 1), strides=(1, 1), rngs=rngs)
        self.bn1 = nnx.BatchNorm(features, use_running_average=not train, rngs=rngs)
        self.conv2_spatial = nnx.Conv(in_features=features, out_features=features,
                                      kernel_size=(2, 2), padding='VALID', rngs=rngs)
        self.bn2 = nnx.BatchNorm(features, use_running_average=not train, rngs=rngs)
        self.conv3_spatial = nnx.Conv(in_features=features, out_features=features,
                                      kernel_size=(2, 2), strides=(1, 1), padding='VALID',
                                      rngs=rngs)
        self.bn3 = nnx.BatchNorm(features, use_running_average=not train, rngs=rngs)
        final_spatial_dim = conv_output_size(
            conv_output_size(patch_size, 2, 1), 2, 1
        )
        input_to_mlp_size = final_spatial_dim * final_spatial_dim * features  # 2 * 2 * features
        self.linear1 = nnx.Linear(in_features=input_to_mlp_size, out_features=mlp_hidden_size,
                                  rngs=rngs)
        self.linear2 = nnx.Linear(in_features=mlp_hidden_size, out_features=2, rngs=rngs)

    def __call__(self, patch1: jnp.ndarray, patch2: jnp.ndarray, prior: jnp.ndarray):
        assert patch1.shape == patch2.shape
        pshape = patch1.shape
        non_batch_dimensions = pshape[-3:]
        batch_dimensions = pshape[:-3]
        patch1 = patch1.reshape(-1, *non_batch_dimensions)
        patch2 = patch2.reshape(-1, *non_batch_dimensions)
        x = jnp.concatenate([patch1, patch2], axis=-1)
        prior_reshaped = prior.reshape(-1, 1, 1, 2)
        prior_tiled = jnp.tile(prior_reshaped, (1, x.shape[1], x.shape[2], 1))
        x = jnp.concatenate([x, prior_tiled], axis=-1)
        x = self.conv1_1x1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        x = self.conv2_spatial(x)
        x = self.bn2(x)
        x = nnx.relu(x)
        x = self.conv3_spatial(x)
        x = self.bn3(x)
        x = nnx.relu(x)
        batch_size = x.shape[0]
        flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.reshape(batch_size, flattened_size)
        x = self.linear1(x)
        x = nnx.relu(x)
        output = self.linear2(x)
        output = output.reshape(*batch_dimensions, 2)
        output = jax.nn.tanh(output)
        return output


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
