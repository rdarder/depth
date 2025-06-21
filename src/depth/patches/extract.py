import jax
import jax.numpy as jnp

from depth.images.separable_convolution import conv_output_size


def extract_patches_nhwc(img: jax.Array, patch_size: int, stride: int) -> jax.Array:
    """Extract patches from an image following a convolution stride pattern.

    img shape: Batch, Height, Width, Channel
    return shape: Batch, Patches_Y, Patches_X, PatchSize, PatchSize, Channel,

    Note that the current implementation returns square patches, but if needed it could be
    simply modified to return rectangular patches.
    """
    B, H, W, C = img.shape
    PY = conv_output_size(H, patch_size, stride)
    PX = conv_output_size(W, patch_size, stride)
    depth_patches = jax.lax.conv_general_dilated_patches(
        img, (patch_size, patch_size), (stride, stride), padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC')
    )
    patches_channel_last = (
        depth_patches.reshape(B, PY, PX, C, patch_size, patch_size)
        .transpose(0, 1, 2, 4, 5, 3)
    )
    return patches_channel_last


def test_minimal_extract_patches():
    imgs = jax.random.uniform(jax.random.key(1), (2, 2, 2, 2))
    patches = extract_patches_nhwc(imgs, patch_size=1, stride=1)
    img_from_patches = patches.reshape(2, 2, 2, 2)
    assert jnp.all(img_from_patches == imgs)


def test_extract_3x3_stride_2():
    img = jax.random.uniform(jax.random.key(1), (1, 5, 7, 1))
    patches = extract_patches_nhwc(img, patch_size=3, stride=2)
    assert patches.shape == (1, 2, 3, 3, 3, 1)
    assert jnp.allclose(
        patches[0, 3, 1], img[0, 2:5, 2:5, :]
    )
    assert jnp.allclose(
        patches[0, 3, 2], img[0, 2:5, 4:7, :]
    )
