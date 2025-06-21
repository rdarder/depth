from __future__ import annotations

import jax
from flax import nnx
from flax.nnx import Rngs
from jax import numpy as jnp

from depth.images.separable_convolution import conv_output_size
from depth.patches.extract import extract_patches_nhwc


class PatchFlowEstimator(nnx.Module):
    """
    A small neural network for estimating optical flow on patches.

    Args:
        num_channels: Number of channels in input patches (e.g., 1 for grayscale, 3 for RGB).
        features_dim: Number of feature maps after the first convolutional layer.
        rngs: An nnx.Rngs object for parameter initialization.
    """

    def __init__(self, patch_size: int, num_channels: int, features_dim: int, *, rngs: Rngs):
        assert patch_size >= 3
        self.num_channels = num_channels
        self.features_dim = features_dim
        self.patch_size = patch_size

        self.feat_conv = nnx.Conv(
            in_features=num_channels,
            out_features=features_dim,
            kernel_size=(patch_size - 1, patch_size - 1),
            padding='VALID',
            rngs=rngs,
        )
        feat_conv_output = conv_output_size(self.patch_size, self.patch_size - 1, 1)

        self.cost_volume_conv = nnx.Conv(
            in_features=features_dim * 2,
            out_features=features_dim,
            kernel_size=(patch_size - 2, patch_size - 2),
            padding='VALID',
            rngs=rngs,
        )
        vol_conv_output = conv_output_size(feat_conv_output, self.patch_size - 2, 1)
        mlp_input_size = vol_conv_output * vol_conv_output * features_dim + 2
        self.mlp_hidden = nnx.Linear(
            in_features=mlp_input_size,
            out_features=mlp_input_size // 2,
            rngs=rngs,
        )
        self.flow_output = nnx.Linear(
            in_features=mlp_input_size // 2,
            out_features=2,
            rngs=rngs,
        )

    def __call__(self, patch1_grid: jax.Array, patch2_grid: jax.Array,
                 priors_grid: jax.Array) -> jax.Array:
        # patch1: NYXHWC
        # priors: N2

        # for shape docs, assume P is the original patch size, P1 is the output size of
        # the first convolution (is 2 for 4x4 patches and 3x3 kernel) and P2 the output
        # of the second convolution (is 1 for the above example). larger patches will
        # yield different intermediate sizes.

        N, PY, PX, H, W, C = patch1_grid.shape
        assert H == W == self.patch_size
        assert patch1_grid.shape == patch2_grid.shape
        B = N * PY * PX

        patch1 = patch1_grid.reshape(B, H, W, C)
        patch2 = patch2_grid.reshape(B, H, W, C)
        priors = priors_grid.reshape(B, 2)

        combined_patches = jnp.concatenate(
            [patch1, patch2], axis=0
        )  # (2*B, P, P, C)

        features = self.feat_conv(combined_patches)
        features = nnx.relu(
            features
        )  # Output: (2*B, P1, P1, features_dim)

        patch1_features, patch2_features = jnp.split(
            features, 2, axis=0
        )  # each (B, P1, P1, features_dim)

        diff_features = jnp.abs(patch1_features - patch2_features)
        prod_features = patch1_features * patch2_features

        combined_feat_diff_prod = jnp.concatenate(
            [diff_features, prod_features], axis=-1
        )  # Shape: (B, P1, P1, 2*features_dim)

        cost_volume_processed = self.cost_volume_conv(combined_feat_diff_prod)
        cost_volume_processed = nnx.relu(
            cost_volume_processed
        )  # Output: B, P2, P2, features_dim)

        cost_volume_flat = cost_volume_processed.reshape(
            B, -1
        )  # (B, P2*P2*features_dim)

        combined_input_to_mlp = jnp.concatenate(
            [cost_volume_flat, priors], axis=-1
        )  # B, (P2)*(P2)*features_dim + 2

        x = self.mlp_hidden(combined_input_to_mlp)
        x = nnx.relu(x)  # (B, ((P2)*(P2)*features_dim + 2) / 2)
        flow_delta = self.flow_output(x)  # Shape: (B, 2)
        flow_delta_grid = flow_delta.reshape(N, PY, PX, 2)

        return flow_delta_grid


def test_patch_flow_estimator():
    rngs = nnx.Rngs(0)
    estimator = PatchFlowEstimator(patch_size=4, num_channels=2, features_dim=8, rngs=rngs)
    canvas = jax.random.uniform(jax.random.key(0), (2, 7, 9, 2))  # B,H,W,C
    frame1 = canvas[:, 1:, 1:, :]
    frame2 = canvas[:, :-1, :-1, :]
    patches1 = extract_patches_nhwc(frame1, patch_size=4, stride=2)
    patches2 = extract_patches_nhwc(frame2, patch_size=4, stride=2)
    priors = jnp.zeros((2, 2, 3, 2))
    flow_delta = estimator(patches1, patches2, priors)
    assert flow_delta.shape == priors.shape
