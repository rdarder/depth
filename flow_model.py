from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from flow import estimate_flow_incrementally
from predictor import MinimalPredictor
from pyramid import BaselinePyramid


class OpticalFlow(nnx.Module):
    def __init__(
        self,
        num_pyramid_levels: int = 4,
        pyramid_patch_size: int = 3,
        pyramid_output_channels: int = 4,
        predictor_hidden_features: tuple[int,...] = (32,),
        *,
        rngs: nnx.Rngs,
    ):
        self.pyramid = BaselinePyramid(
            patch_size=pyramid_patch_size,
            num_levels=num_pyramid_levels,
            out_channels=pyramid_output_channels,
            rngs=rngs,
        )
        self.predictor = MinimalPredictor(
            input_features=pyramid_output_channels * 2 + 2,
            hidden_features=predictor_hidden_features,
            rngs=rngs,
        )
        # self.num_pyramid_levels = num_pyramid_levels

    def __call__(
        self, frame1: jax.Array, frame2: jax.Array, priors: Optional[jax.Array]
    ) -> jax.Array:
        """
        Performs full optical flow estimation.
        Args:
            frame1: Batch of first frames [B, H, W, 1]
            frame2: Batch of second frames [B, H, W, 1]
        Returns:
            predicted_flow_level0: Flow at the finest level [B, H, W, 2]
        """
        assert frame1.shape == frame2.shape
        f1_pyramid = self.pyramid(frame1)
        f2_pyramid = self.pyramid(frame2)
        B, H, W, C = f1_pyramid[0].shape
        if priors is None:
            priors = jnp.zeros((B, H * W, 2))
        assert priors.shape == (B, H * W, 2)
        assert priors.dtype == jnp.float32
        return (
            f1_pyramid,
            f2_pyramid,
            estimate_flow_incrementally(
                self.predictor, frame1, frame2, f1_pyramid, f2_pyramid, priors
            ),
        )
