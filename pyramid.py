# Potential code sketch based on the refined plan
import flax.nnx as nnx
import jax
import jax.numpy as jnp


class AverageFixedConv(nnx.Module):
    def __init__(self, patch_size: int, strides: int):
        self.patch_size = patch_size
        self.strides = strides

    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        assert C == 1
        sum = jax.lax.reduce_window(
            x,
            0.0,
            jax.numpy.add,
            (1, self.patch_size, self.patch_size, 1),
            (1, self.strides, self.strides, 1),
            padding="same",
        )
        return sum / (self.patch_size**2)


class BaselinePyramid(nnx.Module):
    """
    Learned Feature Pyramid using a shared convolution.

    Generates a sequence of feature maps at decreasing resolutions.
    """

    def __init__(
        self,
        patch_size: int = 3,
        strides: int = 2,
        num_levels: int = 4,
        out_channels: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        # Standard practice to pass rngs
        self.num_levels = num_levels
        # Define the shared convolution layer. Parameters are initialized here.
        # in_channels=1 because the input to the conv is always a single channel image/feature map.
        # out_channels=4 as specified.
        self.shared_conv = nnx.Conv(
            in_features=1,
            out_features=out_channels-1,
            kernel_size=(patch_size, patch_size),
            strides=strides,
            padding="SAME",  # Or 'VALID', depending on desired behaviour at edges
            # kernel_init=nnx.initializers.lecun_normal(), # Example initializer
            # bias_init=nnx.initializers.zeros(), # Example initializer
            rngs=rngs,  # Pass rngs for initialization if needed by init functions
        )
        self.avg_conv = AverageFixedConv(patch_size, strides)
        # Note: self.shared_conv is an nnx.Module, its parameters are nested inside self.

    def __call__(self, x: jax.Array) -> list[jax.Array]:
        """
        Applies the pyramid generation process.

        Args:
            x: Input image, expected shape [B, H, W, 1] (grayscale).

        Returns:
            A list of feature maps, where level_features[i] is the 4-channel
            output of the shared_conv at the i-th pyramid level (features
            for Level i+1). The list will have num_levels elements.
        """
        level_features = []

        for _ in range(self.num_levels):
            # Apply the shared convolution to the current level's input
            conv_output = self.shared_conv(x)  # Shape: [B, H/2, W/2, 4]
            x = self.avg_conv(x)

            # Store the 4-channel output features for this level
            level_out = jnp.concatenate([x, conv_output], axis=-1)
            level_features.append(level_out)

        return level_features[::-1]


def example_usage():
    # Example Usage (outside the class definition)
    # Needs a PRNG key
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(params=key)  # Often params collection needs keys

    # Create a dummy input image (batch size 1, 64x64, 1 channel)
    dummy_input = jnp.zeros((1, 64, 64, 1))

    # Instantiate the module
    pyramid_module = BaselinePyramid(num_levels=3, rngs=rngs)

    # The module instance holds the parameters
    print("Pyramid module state:", nnx.split(pyramid_module, nnx.Param))

    # Run the forward pass
    features = pyramid_module(dummy_input)

    print(f"Generated {len(features)} feature levels")
    for i, feats in enumerate(features):
        print(f"Level {i + 1} features shape: {feats.shape}")


if __name__ == "__main__":
    example_usage()
