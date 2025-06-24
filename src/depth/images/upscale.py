import jax
import jax.numpy as jnp

from depth.images.separable_convolution import separable_convolution


def upscale_values_2n_plus2(flow: jax.Array) -> jax.Array:
    """Upscale values of flow, typically done before or after rescaling the flow dimensions.

    The value rescale is such that a value, representing a fraction of the image dimension,
    keeps its proportion after dimension upscaling. In this case we upscale by "2n+2" which is
    the opposite of what we reduced when we constructed an image pyramid with a 4 sized kernel.
    """
    B, H, W, C = flow.shape
    upscale_ratio = (2 * H + 2) / H
    return upscale_ratio * flow


def upscale_size_2n_plus_2(flow: jax.Array) -> jax.Array:
    """Upscales images of shape (B, H, W, C) to (B, 2*H+2, 2W+2, C) and smooths it afterwards."""
    extended_upscaled_flow = flow.repeat(2, axis=1).repeat(2, axis=2)
    extended_padded_upscaled_flow = jnp.pad(
        extended_upscaled_flow, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='edge'
    )
    kernel = jnp.array([[[0.3, 1, 0.3]]]).astype(jnp.float32)
    kernel = kernel / jnp.sum(kernel)
    smooth_upscaled_flow = separable_convolution(
        extended_padded_upscaled_flow, kernel, stride=1
    )
    return smooth_upscaled_flow


def test_upscale_flow():
    sample_flow = jnp.arange(8).reshape(2, 2, 2, 1).astype(jnp.float32)
    upscaled = upscale_size_2n_plus_2(sample_flow)
    assert upscaled.shape == (2, 6, 6, 1)


def test_upscale_values():
    sample_flow = jnp.arange(8).reshape(2, 2, 2, 1).astype(jnp.float32)
    upscaled_values = upscale_values_2n_plus2(sample_flow)
    expected_ratio = (2 * 2 + 2) / 2
    assert jnp.all((upscaled_values / expected_ratio) == sample_flow)
