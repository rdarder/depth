from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from images.separable_convolution import separable_convolution, conv_output_size
from .datasets import load_frame_from_path


class ImageDownscaler(nnx.Module):
    def __init__(self, alpha: float = 0.5, stride: int = 2):
        self.alpha = nnx.Param(alpha)
        self.stride = stride

    def __call__(self, imgs: jax.Array):
        kernel = jnp.array([self.alpha, 1, 1, self.alpha]).reshape(1, 1, 4)
        kernel = kernel / jnp.sum(kernel)
        return separable_convolution(imgs, kernel, self.stride)


class ImagePyramidDecomposer(nnx.Module):
    def __init__(self, downscaler: ImageDownscaler):
        self._downscaler = downscaler

    def __call__(self, img: jax.Array, levels: int) -> Sequence[jax.Array]:
        C, H, W = img.shape
        current = img[None, :, :, :]
        pyramid = [current]
        for level in range(levels):
            current = self._downscaler(current)
            pyramid.append(current)
        return pyramid


def test_pyramid_sizes():
    stride = 2
    downscaler = ImageDownscaler(alpha=-0.2, stride=stride)
    pyramid_decomposer = ImagePyramidDecomposer(downscaler)
    current_size = 100
    frame = load_frame_from_path(
        'test_fixtures/frame1.png', current_size
    )
    kernel_size = 4
    levels = 4
    pyramid = pyramid_decomposer(frame, levels=levels)
    for img in pyramid:
        assert img.shape == (1, 1, current_size, current_size)
        current_size = conv_output_size(current_size, kernel_size, stride)


def plot_pyramid():
    import matplotlib.pyplot as plt
    downscaler = ImageDownscaler(alpha=0.2, stride=2)
    pyramid_decomposer = ImagePyramidDecomposer(downscaler)
    frame = load_frame_from_path(
        'test_fixtures/frame1.png', 158
    )
    pyramid = pyramid_decomposer(frame, levels=4)
    fig, axes = plt.subplots(1, len(pyramid), figsize=(4 * len(pyramid), 4))
    for img, axe in zip(pyramid, axes):
        axe.imshow(img.squeeze((0, 1)), vmin=0, vmax=1, cmap='gray')
    plt.show()


if __name__ == '__main__':
    plot_pyramid()
