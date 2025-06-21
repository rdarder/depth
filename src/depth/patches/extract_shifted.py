import jax
import jax.numpy as jnp

from depth.images.separable_convolution import conv_output_size
from depth.images.valid_grid import get_valid_offsets_flat


def extract_single_shifted_patch(img: jax.Array, offset: jax.Array, patch_size: int):
    """
    img: H, W, C
    start_yx: 2,
    patch_size: ()
    returns: patch_size, patch_size, C
    """
    H, W, C = img.shape
    assert offset.shape == (2,)
    patch = jax.lax.dynamic_slice(
        img,
        (offset[0], offset[1], 0),
        (patch_size, patch_size, C),
        allow_negative_indices=False
    )
    return patch


def extract_shifted_patches_nchw(frame, frame_int_flow, patch_size: int, stride: int):
    """ Extract patches of size patch_size from an image given a flow map.

    The flow is to be interpreted as: the flow entry i,j determines that the patch will be
    taken from img[i*stride + flow_y: +patch_size, j*stride+flow_x: +patch_size]
    Works on a single frame, single_frame: C, H, W
    frame_int_flow: 2, conv_output_size(H, patch_size, stride),
                       conv_output_size(W, patch_size, stride)
    Returns the extracted patches and a grid of booleans telling whether the patch was within
    bounds. When the patch was not within bounds, the patch will still be returned but
    clamped according to jax.lax.dynamic_slice behavior.
        C * (H-patch_size+1) * (W-patch_size+1), patch_size, patch_size,
        C * (H-patch_size+1) * (W-patch_size+1), 1 (bool)
    """
    H, W, C = frame.shape
    FH, FW, F = frame_int_flow.shape
    assert F == 2
    assert FH == conv_output_size(H, patch_size, stride)
    assert FW == conv_output_size(W, patch_size, stride)
    grid_pair = jnp.meshgrid(
        jnp.arange(FH), jnp.arange(FW), indexing='ij'
    )
    grid = jnp.stack(grid_pair, axis=-1) * stride
    offsets_grid = grid + frame_int_flow
    offsets_grid_flat = offsets_grid.reshape(-1, 2)  # (FH*FW, 2)
    valid_coords_flat = get_valid_offsets_flat(offsets_grid_flat, H, W, patch_size)

    extract_all_patches = jax.vmap(extract_single_shifted_patch, in_axes=(None, 0, None))
    patches = extract_all_patches(frame, offsets_grid_flat, patch_size)
    return (
        patches.reshape(FH, FW, patch_size, patch_size, C),
        valid_coords_flat.reshape(FH, FW)
    )


batch_extract_shifted_patches_nchw = jax.vmap(extract_shifted_patches_nchw,
                                              in_axes=(0, 0, None, None))


def test_extract_single_patch():
    canvas = jax.random.uniform(jax.random.key(1), (3, 4, 2))
    patch = extract_single_shifted_patch(canvas, jnp.array([1, 2]), 2)
    expected = canvas[1:3, 2:4, :]
    assert jnp.allclose(patch, expected)


def test_extract_shifted_patches():
    canvas = jnp.arange(32).reshape(4, 4, 2).astype(jnp.float32)
    flow = jnp.array([
        [[0, 0], [-1, 0]],
        [[1, 1], [-1, 0]]
    ])

    patches, valid_grid = extract_shifted_patches_nchw(canvas, flow, 2, 2)
    assert jnp.all(patches[0, 0, :, :, 0] == canvas[0:2, 0:2, 0])
    assert jnp.all(patches[1, 1, :, :, 0] == canvas[1:3, 2:4, 0])
    assert jnp.all(valid_grid == jnp.array([[True, False], [False, True]]))
