import jax
import jax.numpy as jnp
import flax.linen as nn

class MinimalPredictor(nn.Module):
    """
    A minimal MLP predictor for per-pixel motion from concatenated features.
    It is applied point-wise across spatial dimensions.
    Input: (batch_size, height, width, 8) - 4 features from F1, 4 from F2
    Output: (batch_size, height, width, 2) - flow vector (ux, uy)
    """
    hidden_features: int = 16 # Number of units in the hidden layer

    @nn.compact
    def __call__(self, x):
        """
        Applies the MLP predictor.

        Args:
            x: Input tensor of shape (batch_size, height, width, 8).
               Can also be (batch_size, 8) if processing single points per batch item.

        Returns:
            Output tensor of shape (batch_size, height, width, 2) or (batch_size, 2).
        """
        # The input tensor `x` is expected to have the feature dimension as the last one.
        # Flax's Dense layers operate on the last dimension and broadcast over others.

        # Hidden layer: Dense + ReLU
        # Input features: 8
        # Output features: self.hidden_features
        x = nn.Dense(features=self.hidden_features, name="hidden_layer")(x)
        x = nn.relu(x) # Apply ReLU activation

        # Output layer: Dense (linear activation for flow components)
        # Input features: self.hidden_features
        # Output features: 2 (for ux, uy)
        x = nn.Dense(features=2, name="output_layer")(x)

        return x

if __name__ == '__main__':
    # Example Usage:
    key = jax.random.PRNGKey(1) # Use a PRNG key for parameter initialization

    # Create a dummy input representing concatenated features from a single level
    # Let's simulate a batch of 1 image, at level 3 (e.g., original 256x256 -> 32x32)
    # Input shape: (batch_size, H_k, W_k, 8)
    dummy_predictor_input_grid = jnp.ones((1, 32, 32, 8), dtype=jnp.float32)

    # Initialize the predictor module
    predictor_model = MinimalPredictor(hidden_features=16) # Using the default 16 hidden units

    # Initialize parameters of the module
    # Flax initialization requires a PRNG key and an example input shape/dtype
    variables = predictor_model.init(key, dummy_predictor_input_grid)
    params = variables['params'] # Extract the learned parameters

    print("--- Grid Input Example ---")
    print(f"Predictor input shape: {dummy_predictor_input_grid.shape}")

    # Apply the model (forward pass) to the grid of features
    predicted_flow_grid = predictor_model.apply({'params': params}, dummy_predictor_input_grid)

    print(f"Predicted flow shape: {predicted_flow_grid.shape}") # Should be (1, 32, 32, 2)

    print("\n--- Single Pixel Input Example ---")
    # Create a dummy input representing features from a *single* spatial location
    # Input shape: (batch_size, 8) - batch size could be many points from one level
    dummy_single_pixel_input = jnp.arange(8).reshape(1, 8).astype(jnp.float32)
    # Or multiple pixels in a batch:
    dummy_multiple_pixels_input = jnp.stack([jnp.arange(8), jnp.arange(8) * 2], axis=0).astype(jnp.float32)
    print(f"Single pixel input shape: {dummy_single_pixel_input.shape}")
    print(f"Multiple pixels input shape: {dummy_multiple_pixels_input.shape}")


    # Apply the model to the single pixel input
    predicted_flow_single_pixel = predictor_model.apply({'params': params}, dummy_single_pixel_input)
    print(f"Predicted flow single pixel shape: {predicted_flow_single_pixel.shape}") # Should be (1, 2)

    # Apply the model to the multiple pixels input
    predicted_flow_multiple_pixels = predictor_model.apply({'params': params}, dummy_multiple_pixels_input)
    print(f"Predicted flow multiple pixels shape: {predicted_flow_multiple_pixels.shape}") # Should be (2, 2)