import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from numpy.testing import assert_array_equal

from flow import estimate_flow_at_level, gather_frame_values
from predictor import MinimalPredictor


class TestGatherFrameValues(unittest.TestCase):
    def test_gather_frame_values_single_batch(self):
        """Tests gather_frame_values with a single batch item."""
        frame = jnp.array(
            [[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=jnp.int32
        )  # Shape (1, 3, 3, 1)

        # Coordinates to gather: (0,0), (1,1), (2,2), (1,2)
        coords = jnp.array(
            [[[0, 0], [1, 1], [2, 2], [1, 2]]], dtype=jnp.int32
        )  # Shape (1, 4, 2)

        expected_gathered = jnp.array(
            [[[1], [5], [9], [6]]], dtype=jnp.int32
        )  # Shape (1, 4, 1)

        gathered = gather_frame_values(frame, coords)

        assert_array_equal(gathered, expected_gathered)
        assert gathered.shape == expected_gathered.shape

    def test_gather_frame_values_multiple_batches(self):
        """Tests gather_frame_values with multiple batch items."""
        frame = jnp.array(
            [
                [[[10, 11], [12, 13]], [[14, 15], [16, 17]]],  # Batch 0 (2x2x2)
                [[[20, 21], [22, 23]], [[24, 25], [26, 27]]],  # Batch 1 (2x2x2)
            ],
            dtype=jnp.int32,
        )  # Shape (2, 2, 2, 2)

        # Coordinates to gather:
        # Batch 0: (0,0), (1,1)
        # Batch 1: (0,1), (1,0)
        coords = jnp.array(
            [[[0, 0], [1, 1]], [[0, 1], [1, 0]]], dtype=jnp.int32
        )  # Shape (2, 2, 2)

        expected_gathered = jnp.array(
            [[[10, 11], [16, 17]], [[22, 23], [24, 25]]], dtype=jnp.int32
        )  # Shape (2, 2, 2)

        gathered = gather_frame_values(frame, coords)

        assert_array_equal(gathered, expected_gathered)
        assert gathered.shape == expected_gathered.shape


class TestFlowLevelEstimation(unittest.TestCase):
    def test_estimate_flow_at_level_shapes(self):
        """
        Tests estimate_flow_at_level for shape consistency, highlighting
        the potential shape error during concatenation due to priors' shape.
        """
        B, H, W, C = (
            1,
            10,
            10,
            3,
        )  # Batch, Height, Width, Channels for images and priors
        N = 5  # Number of coordinate points selected by gather_frame_values

        # Create dummy inputs with shapes based on the function's expectations and assertions
        f1 = jnp.ones((B, H, W, C))
        f2 = jnp.ones((B, H, W, C))
        priors = jnp.ones((B, H, W, C))  # Priors shape matches f1/f2 as per assertion

        # Coords have shape (B, N, 2)
        f1_coords = jnp.zeros((B, N, 2), dtype=jnp.int32)
        f2_coords = jnp.zeros((B, N, 2), dtype=jnp.int32)

        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)
        mock_predictor = MinimalPredictor(rngs=rngs)

        # The concatenation step is expected to raise a shape error
        # because f1_selected/f2_selected have shape (B, N, C) and priors has shape (B, H, W, C).
        # N is not equal to H*W, so the dimensions before the last one won't match for concatenation.
        with self.assertRaises(TypeError):
            estimate_flow_at_level(mock_predictor, f1, f2, f1_coords, f2_coords, priors)
