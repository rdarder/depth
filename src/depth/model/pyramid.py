from importlib import resources
from typing import Sequence

import jax
from flax import nnx
from jax import numpy as jnp

from depth.images.load import load_frame_from_path
from depth.images.separable_convolution import depthwise_separable_convolution, conv_output_size


class ImageDownscaler:
    def __init__(self, alpha: float = 0.5, stride: int = 2):
        self.alpha = nnx.Param(alpha)
        self.stride = stride

    def __call__(self, imgs: jax.Array):
        kernel = jnp.array([self.alpha, 1, 1, self.alpha]).reshape(1, 1, 4)
        kernel = kernel / jnp.sum(kernel)
        return depthwise_separable_convolution(imgs, kernel, self.stride)


class ImagePyramidDecomposer:
    def __init__(self, downscaler: ImageDownscaler, levels: int):
        self._downscaler = downscaler
        self.levels = levels

    def __call__(self, img: jax.Array) -> Sequence[jax.Array]:
        N, H, W, C = img.shape
        current = img
        pyramid = [current]
        for level in range(self.levels):
            current = self._downscaler(current)
            pyramid.append(current)
        return pyramid


def test_pyramid_sizes():
    stride = 2
    downscaler = ImageDownscaler(alpha=-0.2, stride=stride)
    pyramid_decomposer = ImagePyramidDecomposer(downscaler, levels=4)
    current_size = 100
    frame_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame = load_frame_from_path(str(frame_path), current_size)
    kernel_size = 4
    pyramid = pyramid_decomposer(frame[None, :, :, :])
    for img in pyramid:
        assert img.shape == (1, current_size, current_size, 1)
        current_size = conv_output_size(current_size, kernel_size, stride)


def plot_pyramid():
    import matplotlib.pyplot as plt
    downscaler = ImageDownscaler(alpha=0.2, stride=2)
    pyramid_decomposer = ImagePyramidDecomposer(downscaler, levels=4)
    frame_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame = load_frame_from_path(str(frame_path), 158)
    pyramid = pyramid_decomposer(frame[None, :, :, :])
    fig, axes = plt.subplots(1, len(pyramid), figsize=(4 * len(pyramid), 4))
    for img, axe in zip(pyramid, axes):
        axe.imshow(img.squeeze((0, 1)), vmin=0, vmax=1, cmap='gray')
    plt.show()


if __name__ == '__main__':
    plot_pyramid()
