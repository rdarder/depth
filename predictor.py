import jax
import jax.numpy as jnp
import flax.nnx as nnx

class MinimalPredictor(nnx.Module):
    """
    Minimal learned per-pixel motion predictor (MLP) with flow prior.

    Takes concatenated features from two frames at the same location,
    plus a 2D flow prior, and predicts a 2D flow residual.
    """
    def __init__(self, hidden_features: int = 32, *, rngs: nnx.Rngs):
        # Input features:
        #   4 from frame1 features
        # + 4 from frame2 features
        # + 2 from flow prior (prior_ux, prior_uy)
        # = 10 total input features
        #
        # Output features: 2 (delta_ux, delta_uy) - a residual to the prior
        self.dense1 = nnx.Linear(
            in_features=10, # Updated input dimension
            out_features=hidden_features,
            rngs=rngs
        )
        self.dense2 = nnx.Linear(
            in_features=hidden_features,
            out_features=2, # Outputting a 2D residual
            rngs=rngs
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Applies the MLP prediction.

        Args:
            inputs: Concatenated features and prior for a batch of locations,
                    expected shape [B', 10].
                    The last 2 elements of the 10 are (prior_ux, prior_uy).

        Returns:
            Predicted flow residuals (delta_ux, delta_uy), shape [B', 2].
        """
        x = self.dense1(inputs)
        x = jax.nn.relu(x)
        residuals = self.dense2(x)
        return residuals

def sample_usage():
    # Example Usage (outside the class definition)
    # Needs a PRNG key
    key = jax.random.PRNGKey(1)
    rngs = nnx.Rngs(params=key)

    # Create dummy input (batch size 10, 10 features each)
    dummy_features_f1 = jnp.zeros((10, 4))
    dummy_features_f2 = jnp.zeros((10, 4))
    dummy_prior_flow = jnp.ones((10, 2)) * 0.5 # e.g., some non-zero prior

    # Concatenate inputs:
    dummy_input_batch_with_prior = jnp.concatenate(
        [dummy_features_f1, dummy_features_f2, dummy_prior_flow], axis=-1
    )
    assert dummy_input_batch_with_prior.shape == (10, 10)

    # Instantiate the module
    predictor_module_with_prior = MinimalPredictor(hidden_features=32, rngs=rngs)

    # The module instance holds the parameters
    print("Predictor module (with prior) state:", nnx.split(predictor_module_with_prior, nnx.Param))

    # Run the forward pass
    predicted_residuals = predictor_module_with_prior(dummy_input_batch_with_prior)

    print(f"Input batch shape: {dummy_input_batch_with_prior.shape}")
    print(f"Predicted residuals shape: {predicted_residuals.shape}")

    # If these were actual predictions, the flow for these locations would be:
    final_flow_at_locations = dummy_prior_flow + predicted_residuals
    print(f"Final flow shape: {final_flow_at_locations.shape}")


if __name__ == "__main__":
    sample_usage()
