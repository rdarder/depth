import flax.nnx as nnx
import jax

from incremental import hierarchical_flow_estimation
from loss import compute_photometric_loss
from predictor import MinimalPredictor
from pyramid import BaselinePyramid


class OpticalFlow(nnx.Module):
    def __init__(
        self, num_pyramid_levels: int, predictor_hidden_features: int, *, rngs: nnx.Rngs
    ):
        self.pyramid = BaselinePyramid(num_levels=num_pyramid_levels, rngs=rngs)
        self.predictor = MinimalPredictor(
            hidden_features=predictor_hidden_features, rngs=rngs
        )
        self.num_pyramid_levels = num_pyramid_levels

    def __call__(self, frame1: jax.Array, frame2: jax.Array) -> jax.Array:
        """
        Performs full optical flow estimation.
        Args:
            frame1: Batch of first frames [B, H, W, 1]
            frame2: Batch of second frames [B, H, W, 1]
        Returns:
            predicted_flow_level0: Flow at the finest level [B, H, W, 2]
        """
        frame1_features = self.pyramid(frame1)
        frame2_features = self.pyramid(frame2)

        predicted_flow_level0 = hierarchical_flow_estimation(
            frame1_features,
            frame2_features,
            self.predictor,  # Pass the predictor instance
            self.num_pyramid_levels,
        )
        return predicted_flow_level0


def loss_fn_for_grad(
    model: OpticalFlow,
    batch_frame1: jax.Array,
    batch_frame2: jax.Array,
    patch_size: int,
):
    """Computes the loss for gradient calculation."""
    # Forward pass through the main model
    predicted_flow = model(batch_frame1, batch_frame2)

    # Compute photometric loss
    loss = compute_photometric_loss(
        batch_frame1,
        batch_frame2,
        predicted_flow,
        patch_size=patch_size,
        loss_type="l1",  # Or 'l2'
    )

    return loss
