import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def apply_flow_entire_image(img: jax.Array, flow: jax.Array) -> jax.Array:
    flow_y = flow[:, :, 0]
    flow_x = flow[:, :, 1]
    H, W = flow_y.shape
    grid_y, grid_x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    return map_coordinates(img, [grid_y + flow_y, grid_x + flow_x], order=1)


apply_flow_multi_channel_image = jax.vmap(apply_flow_entire_image, in_axes=(-1, None), out_axes=-1)
