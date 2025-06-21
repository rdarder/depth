import jax
import jax.numpy as jnp

from images.separable_convolution import separable_convolution


def upscale_flow_2n_plus_2(flow: jax.Array, rescale_values: bool = True) -> jax.Array:
    B, F, H, W = flow.shape
    assert F == 2
    if rescale_values:
        upscale_ratio = (2 * H + 2) / H
        flow = upscale_ratio * flow  # incorrect
    extended_upscaled_flow = flow.repeat(2, axis=2).repeat(2, axis=3)
    extended_padded_upscaled_flow = jnp.pad(
        extended_upscaled_flow, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='edge'
    )
    kernel = jnp.array([[[0.3, 1, 0.3]]]).astype(jnp.float32)
    kernel = kernel / jnp.sum(kernel)
    smooth_upscaled_flow = separable_convolution(
        extended_padded_upscaled_flow, kernel, stride=1
    )
    return smooth_upscaled_flow
