from importlib import resources

import jax
from jax import numpy as jnp

from depth.images.load import load_frame_from_path
from depth.images.separable_convolution import depthwise_separable_convolution, conv_output_size


def downscale_image(imgs: jax.Array):
    alpha = 0.5
    kernel = jnp.array([alpha, 1, 1, alpha]).reshape(1, 1, 4)
    kernel = kernel / jnp.sum(kernel)
    return depthwise_separable_convolution(imgs, kernel, 2)


def build_image_pyramid(img: jax.Array, levels: int, keep: int):
    assert levels >= 1
    assert 1 <= keep <= levels
    current = img
    pyramid = [current]
    for level in range(levels - 1):
        current = downscale_image(current)
        pyramid.append(current)
    return pyramid[-keep:]


def test_pyramid_sizes():
    stride = 2
    current_size = 100
    frame_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame = load_frame_from_path(str(frame_path), current_size)
    kernel_size = 4
    keep = 2
    levels = 4
    pyramid = build_image_pyramid(frame[None, :, :, :], levels=levels, keep=keep)
    assert len(pyramid) == keep
    for i in range(levels - keep):
        current_size = conv_output_size(current_size, kernel_size, stride)
    for img in pyramid:
        assert img.shape == (1, current_size, current_size, 1)
        current_size = conv_output_size(current_size, kernel_size, stride)


def plot_pyramid():
    import matplotlib.pyplot as plt
    frame_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame = load_frame_from_path(str(frame_path), 158)
    pyramid = build_image_pyramid(frame[None, :, :, :], levels=6, keep=3)
    fig, axes = plt.subplots(1, len(pyramid), figsize=(4 * len(pyramid), 4))
    for img, axe in zip(pyramid, axes):
        axe.imshow(img.squeeze((0, -1)), vmin=0, vmax=1, cmap='gray')
    plt.show()


if __name__ == '__main__':
    plot_pyramid()
