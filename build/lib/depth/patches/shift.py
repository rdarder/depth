import jax
import jax.numpy as jnp

def shifted_center_patch(patch: jax.Array, flow: jax.Array) -> jax.Array:
    """Given an C*H*W patch, returns a hypothetical center patch
    of dimensions C, H-1, W-1 that is off-centered by the given flow.

    When flow is 0.0, then the patch will be an interpolation center
    crop of a patch with size H, W to an H-1, W-1.
    Flow (dy, dx) should not exceed 0.5 in either dimension. This shifting
    is limited to at most half a pixel in either direction.
    """
    C, H, W = patch.shape
    OH = H - 1
    OW = W - 1
    # flow is (dy, dx)
    # kernel should be ([(1-dx)*(1-dy), (dx*(1-dy)], [(1-dx)*dy, dx*dy]])
    neg_flow = 0.5 - flow
    pos_flow = 0.5 + flow
    kernel_y = jnp.array([[[neg_flow[0], pos_flow[0]]]])
    kernel_x = jnp.array([[[neg_flow[1], pos_flow[1]]]])

    patch_for_conv_x = patch.reshape(C * H, 1, W)
    conv_x = jax.lax.conv_general_dilated(
        patch_for_conv_x,
        kernel_x,
        window_strides=(1,),
        padding=((0, 0),),
        dimension_numbers=('NCW', 'IOW', 'NCW'),
    ).reshape(C, H, OW)
    patch_for_conv_y = conv_x.transpose(0, 2, 1).reshape(C * OW, 1, H)
    conv_y = jax.lax.conv_general_dilated(
        patch_for_conv_y,
        kernel_y,
        window_strides=(1,),
        padding=((0, 0),),
        dimension_numbers=('NCH', 'IOH', 'NCH'),
    ).reshape(C, OW, OH).transpose(0, 2, 1)

    return conv_y


def test_shifted_patch_single_channel_top_left():
    sample_patch = jax.random.normal(jax.random.key(1), (1, 3, 5))
    shifted = shifted_center_patch(sample_patch, jnp.array([-0.5, -0.5]))
    assert shifted.shape == (1, 2, 4)
    assert jnp.all(shifted == sample_patch[0, :-1, :-1])


def test_shifted_patch_single_channel_bottom_right():
    sample_patch = jax.random.normal(jax.random.key(1), (1, 3, 5))
    shifted = shifted_center_patch(sample_patch, jnp.array([0.5, 0.5]))
    assert shifted.shape == (1, 2, 4)
    assert jnp.all(shifted == sample_patch[0, 1:, 1:])


def test_shifted_patch_single_channel_bottom_left():
    sample_patch = jax.random.normal(jax.random.key(1), (1, 3, 5))
    shifted = shifted_center_patch(sample_patch, jnp.array([0.5, -0.5]))
    assert shifted.shape == (1, 2, 4)
    assert jnp.all(shifted == sample_patch[0, 1:, :-1])


def test_shifted_patch_single_channel_top_right():
    sample_patch = jax.random.normal(jax.random.key(1), (1, 3, 5))
    shifted = shifted_center_patch(sample_patch, jnp.array([-0.5, 0.5]))
    assert shifted.shape == (1, 2, 4)
    assert jnp.all(shifted == sample_patch[0, :-1, 1:])


def test_shifted_patch_two_channels():
    sample_patch = jax.random.normal(jax.random.key(1), (2, 4, 4))
    shifted = shifted_center_patch(sample_patch, jnp.array([-0.5, -0.5]))
    assert shifted.shape == (2, 3, 3)
    assert jnp.all(shifted == sample_patch[:, :-1, :-1])


def test_shifted_dead_center():
    sample_patch = jnp.arange(9).reshape(1, 3, 3).astype(jnp.float32)
    shifted = shifted_center_patch(sample_patch, jnp.array([0., 0.]))
    expected = jnp.array([[
        0 + 1 + 3 + 4, 1 + 2 + 4 + 5
    ], [
        3 + 4 + 6 + 7, 4 + 5 + 7 + 8
    ]]) / 4.0
    assert jnp.all(shifted == expected)


