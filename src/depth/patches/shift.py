import jax
import jax.numpy as jnp

from depth.images.separable_convolution import separable_convolution

@jax.jit
def shifted_center_patch(patch: jax.Array, flow: jax.Array) -> jax.Array:
    """Given an C*H*W patch, returns a hypothetical center patch
    of dimensions H-1, W-1, C that is off-centered by the given flow.

    When flow is 0.0, then the patch will be an interpolation center
    crop of a patch with size H, W to an H-1, W-1.
    Flow (dy, dx) should not exceed 0.5 in either dimension. This shifting
    is limited to at most half a pixel in either direction.
    """
    dy, dx = flow
    conv_x1 = patch[:, 1:, :] * (0.5 + dx)
    conv_x2 = patch[:, :-1, :] * (0.5 - dx)
    conv_x = conv_x1 + conv_x2
    conv_y1 = conv_x[1:, :, :] * (0.5 + dy)
    conv_y2 = conv_x[:-1, :, :] * (0.5 - dy)
    conv = conv_y1 + conv_y2
    return conv


def test_shifted_patch_single_channel_top_left():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 5, 1))
    shifted = shifted_center_patch(sample_patch, jnp.array([-0.5, -0.5]))
    assert shifted.shape == (2, 4, 1)
    assert jnp.all(shifted == sample_patch[:-1, :-1])


def test_shifted_patch_single_channel_bottom_right():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 5, 1))
    shifted = shifted_center_patch(sample_patch, jnp.array([0.5, 0.5]))
    assert shifted.shape == (2, 4, 1)
    assert jnp.all(shifted == sample_patch[1:, 1:])


def test_shifted_patch_single_channel_bottom_left():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 5, 1))
    shifted = shifted_center_patch(sample_patch, jnp.array([0.5, -0.5]))
    assert shifted.shape == (2, 4, 1)
    assert jnp.all(shifted == sample_patch[1:, :-1])


def test_shifted_patch_single_channel_top_right():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 5, 1))
    shifted = shifted_center_patch(sample_patch, jnp.array([-0.5, 0.5]))
    assert shifted.shape == (2, 4, 1)
    assert jnp.all(shifted == sample_patch[:-1, 1:])


def test_shifted_patch_two_channels():
    sample_patch = jax.random.normal(jax.random.key(1), (4, 4, 2))
    shifted = shifted_center_patch(sample_patch, jnp.array([-0.5, -0.5]))
    assert shifted.shape == (3, 3, 2)
    assert jnp.all(shifted == sample_patch[:-1, :-1])


def test_shifted_dead_center():
    sample_patch = jnp.arange(9).reshape(3, 3, 1).astype(jnp.float32)
    shifted = shifted_center_patch(sample_patch, jnp.array([0., 0.]))
    expected = jnp.array([[
        [0 + 1 + 3 + 4], [1 + 2 + 4 + 5]
    ], [
        [3 + 4 + 6 + 7], [4 + 5 + 7 + 8]
    ]]) / 4.0
    assert jnp.all(shifted == expected)
