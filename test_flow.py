import unittest

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from flow import estimate_flow_at_level, gather_frame_values


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


class MockPredictorCompatible:
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs shape will be (B * N, C + C + P)
        return jnp.ones((inputs.shape[0], 2))


class TestFlowLevelEstimation(unittest.TestCase):
    def test_estimate_flow_at_level_output_shape(self):
        """
        Tests estimate_flow_at_level for correct output shape when inputs are compatible.
        """
        B, H, W, C = 2, 10, 10, 3  # Batch, Height, Width, Channels for images
        N = 5  # Number of coordinate points selected

        # Create dummy inputs with compatible shapes
        f1 = jnp.ones((B, H, W, C))
        f2 = jnp.ones((B, H, W, C))
        # Priors shape is (B, N, P) to be compatible with concatenation after gather_frame_values
        # Note: This deviates from the original assertion f1.shape == priors.shape
        priors = jnp.ones((B, N, 2))

        # Coords have shape (B, N, 2)
        f1_coords = jnp.zeros((B, N, 2), dtype=jnp.int32)
        f2_coords = jnp.zeros((B, N, 2), dtype=jnp.int32)

        mock_predictor = MockPredictorCompatible()

        # Call the function with compatible inputs
        predicted_flow = estimate_flow_at_level(
            mock_predictor, f1, f2, f1_coords, f2_coords, priors
        )

        # Assert the final output shape is (B, N, OutputDim)
        expected_shape = (B, N, 2)
        self.assertEqual(predicted_flow.shape, expected_shape)
