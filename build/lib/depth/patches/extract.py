import jax
import jax.numpy as jnp

from images.separable_convolution import conv_output_size


def extract_patches_nchw(img: jax.Array, patch_size: int, stride: int) -> jax.Array:
    """Extract patches from an image following a convolution stride pattern.

    img shape: Batch, Channel, Height, Width
    return shape: Batch, Channel, Patches_Y, Patches_X, PatchSize, PatchSize

    Note that the current implementation returns squared patches, but if needed it could be
    simply modified to return rectangular patches.
    """
    B, C, H, W = img.shape
    depth_patches = jax.lax.conv_general_dilated_patches(
        img, (patch_size, patch_size), (stride, stride), padding='VALID'
    ).transpose(0, 2, 3, 1)
    PH = conv_output_size(H, patch_size, stride)
    PW = conv_output_size(W, patch_size, stride)
    patches = depth_patches.reshape(B, PH, PW, C, patch_size, patch_size)
    return patches.transpose(0, 3, 1, 2, 4, 5)


def test_extract_unit_patches_unit_stride():
    """The trivial patch extraction should match the input image after a simple reshape"""
    img = jax.random.uniform(jax.random.key(1), (2, 3, 4, 5))
    patches = extract_patches_nchw(img, patch_size=1, stride=1)
    assert jnp.all(img == patches.squeeze((-2, -1)))


def test_extract_3x3_stride_2():
    img = jax.random.uniform(jax.random.key(1), (1, 1, 5, 7))
    patches = extract_patches_nchw(img, patch_size=3, stride=2)
    assert patches.shape == (1, 1, 2, 3, 3, 3)
    assert jnp.all(
        patches[0, 0, 1, 1] == img[:, :, 2:5, 2:5]
    )
    assert jnp.all(
        patches[0, 0, 1, 2] == img[:, :, 2:5, 4:7]
    )
