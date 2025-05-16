import jax
import jax.numpy as jnp
import flax.linen as nn

class BaselinePyramid(nn.Module):
    """
    A pyramid module that iteratively applies a shared convolution.
    Each level outputs 4 feature channels and passes one channel down.
    """

    @nn.compact
    def __call__(self, x):
        """
        Applies the pyramid operation.

        Args:
            x: Input tensor of shape (batch_size, height, width, 1).

        Returns:
            A list of feature tensors, where each tensor F_Lk has shape
            (batch_size, H_orig/(2^k), W_orig/(2^k), 4).
        """

        # Define the shared convolution layer for all levels
        # Takes 1 input channel, outputs 4 feature channels, downsamples by 2
        conv_level = nn.Conv(
            features=4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME', # Ensures output H/2, W/2
            name="shared_conv_level" # Name for parameter sharing
        )

        feature_outputs = []
        current_input = x # Shape (batch, H, W, 1)

        # Loop while the input to the convolution is at least 2x2,
        # ensuring the output is at least 1x1.
        while current_input.shape[1] >= 2 and current_input.shape[2] >= 2:
            level_output_k = conv_level(current_input) # Shape (batch, H/2, W/2, 4)

            feature_outputs.append(level_output_k)

            # Select the first channel to be the input for the next level
            # Slicing to keep the channel dimension: (batch, H/2, W/2, 1)
            current_input = level_output_k[:, :, :, 0:1]

        return feature_outputs

if __name__ == '__main__':
    # Example Usage:
    key = jax.random.PRNGKey(0)

    # Create an example input image (batch_size=1, height=32, width=32, channels=1)
    dummy_image = jnp.ones((1, 32, 32, 1))

    # Initialize the pyramid module
    pyramid_model = BaselinePyramid()

    # Initialize parameters (weights) of the module
    # Flax initialization requires a PRNG key and an example input
    variables = pyramid_model.init(key, dummy_image)
    params = variables['params'] # Extract the learned parameters

    # Apply the model (forward pass)
    # For subsequent calls, use model.apply with the initialized variables
    pyramid_features = pyramid_model.apply({'params': params}, dummy_image)

    print(f"Input image shape: {dummy_image.shape}")
    print(f"Number of pyramid levels generated: {len(pyramid_features)}")
    for i, features_lk in enumerate(pyramid_features):
        print(f"  Level {i+1} (F_L{i+1}) feature shape: {features_lk.shape}")

    # Example with a non-power-of-2 input
    dummy_image_odd = jnp.ones((1, 37, 37, 1))
    pyramid_features_odd = pyramid_model.apply({'params': params}, dummy_image_odd)
    print(f"\nInput image odd shape: {dummy_image_odd.shape}")
    print(f"Number of pyramid levels generated (odd): {len(pyramid_features_odd)}")
    for i, features_lk in enumerate(pyramid_features_odd):
        print(f"  Level {i+1} (F_L{i+1}) feature shape (odd): {features_lk.shape}")
