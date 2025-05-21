import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MinimalPredictor(nnx.Module):
    """
    Minimal learned per-pixel motion predictor (MLP) with flow prior.

    Takes concatenated features from two frames at the same location,
    plus a 2D flow prior, and predicts a 2D flow residual.
    """

    def __init__(
        self, input_features: int = 10, hidden_features: int = 32, *, rngs: nnx.Rngs
    ):
        # Input features:
        #   4 from frame1 features
        # + 4 from frame2 features
        # + 2 from flow prior (prior_ux, prior_uy)
        # = 10 total input features
        #
        # Output features: 2 (delta_ux, delta_uy) - a residual to the prior
        self.dense1 = nnx.Linear(
            in_features=input_features,  # Updated input dimension
            out_features=hidden_features,
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            in_features=hidden_features,
            out_features=3,  # 2D residual flow, 1D confidence
            rngs=rngs,
        )

    def __call__(
        self, f1: jax.Array, f2: jax.Array, priors: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """
        Applies the MLP prediction.

        Args:
            inputs: Concatenated features and prior for a batch of locations,
                    expected shape [B', 10].
                    The last 2 elements of the 10 are (prior_ux, prior_uy).

        Returns:
            Predicted flow residuals (delta_ux, delta_uy) and confidence shape [B', 3].
        """
        inputs = jnp.concatenate([f1, f2, priors], axis=-1)
        x = self.dense1(inputs)
        x = jax.nn.relu(x)
        flow_with_confidence = self.dense2(x)
        flow, confidence = jnp.split(flow_with_confidence, [2], axis=-1)
        return flow_with_confidence


def sample_usage():
    # Example Usage (outside the class definition)
    # Needs a PRNG key
    key = jax.random.PRNGKey(1)
    rngs = nnx.Rngs(params=key)

    # Create dummy input (batch size 10, 10 features each)
    dummy_features_f1 = jnp.zeros((10, 4))
    dummy_features_f2 = jnp.zeros((10, 4))
    dummy_prior_flow = jnp.ones((10, 2)) * 0.5  # e.g., some non-zero prior

    # Concatenate inputs:
    dummy_input_batch_with_prior = jnp.concatenate(
        [dummy_features_f1, dummy_features_f2, dummy_prior_flow], axis=-1
    )
    assert dummy_input_batch_with_prior.shape == (10, 10)

    # Instantiate the module
    predictor_module_with_prior = MinimalPredictor(hidden_features=32, rngs=rngs)

    # The module instance holds the parameters
    print(
        "Predictor module (with prior) state:",
        nnx.split(predictor_module_with_prior, nnx.Param),
    )

    # Run the forward pass
    predicted_residuals = predictor_module_with_prior(dummy_input_batch_with_prior)

    print(f"Input batch shape: {dummy_input_batch_with_prior.shape}")
    print(f"Predicted residuals shape: {predicted_residuals.shape}")
    print(f"Predicted residuals: {predicted_residuals}")

    # If these were actual predictions, the flow for these locations would be:
    final_flow_at_locations = dummy_prior_flow + predicted_residuals
    print(f"Final flow shape: {final_flow_at_locations.shape}")


if __name__ == "__main__":
    sample_usage()
