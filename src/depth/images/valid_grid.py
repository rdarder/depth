import jax
import jax.numpy as jnp


def get_valid_offsets_flat(offsets_flat: jax.Array, H: int, W: int, patch_size: int):
    """Returns which offsets are valid for referencing within a matrix of dimensions H, W.

    Takes a batch B, 2.
    Returns B booleans.
    """
    _, O = offsets_flat.shape
    assert O == 2
    valid_y_min = offsets_flat[:, 0] >= 0
    valid_x_min = offsets_flat[:, 1] >= 0
    valid_y_max = (offsets_flat[:, 0] + patch_size) <= H
    valid_x_max = (offsets_flat[:, 1] + patch_size) <= W
    valid_coords_flat = jnp.logical_and(
        jnp.logical_and(valid_y_min, valid_x_min),
        jnp.logical_and(valid_y_max, valid_x_max)
    )
    return valid_coords_flat


def test_valid_offset_flat():
    offset_grid = jnp.array(
        [[1, 1], [2, 1], [3, 1], [2, 2], [0, 1], [1, 3]], dtype=jnp.float32)
    valid_grid = get_valid_offsets_flat(offset_grid, 3, 4, 2)
    assert valid_grid.shape == (6,)
    assert jnp.all(
        valid_grid == jnp.array([[True, False, False, False, True, False]])
    )
