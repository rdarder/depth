import jax
import jax.numpy as jnp


def depthwise_separable_convolution(imgs: jax.Array, kernel: jax.Array, stride: int):
    N, H, W, C = imgs.shape
    kernel_1d = kernel.flatten()
    kernel_2d_base = jnp.outer(kernel_1d, kernel_1d)
    kernel_for_conv = kernel_2d_base[:, :, None, None].repeat(C, axis=3)
    output = jax.lax.conv_general_dilated(
        imgs,
        kernel_for_conv,
        window_strides=(stride, stride),  # 2D strides
        padding='VALID',  # No padding, matching original
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),  # Standard NHWC input/output, HWIO kernel
        feature_group_count=C  # Apply convolution per input channel
    )
    return output


def conv_output_size(input_size: int, kernel_size: int, stride: int):
    output = (input_size - kernel_size) // stride + 1
    assert output > 0
    return output


def conv_output_size_steps(input_size: int, kernel_size: int, stride: int, steps: int):
    output = input_size
    for i in range(steps - 1):
        output = conv_output_size(output, kernel_size, stride)
    return output


def test_separable_smoothing_filter():
    img = jnp.arange(9).reshape(1, 3, 3, 1).astype(jnp.float32)
    kernel = jnp.array([0.5, 0.5]).reshape(1, 1, 2)
    smooth = depthwise_separable_convolution(img, kernel, stride=1)
    assert smooth.shape == (1, 2, 2, 1)
    assert jnp.all(smooth == jnp.array([[
        [[2.], [3.]], [[5.], [6.]]
    ]]))


def test_separable_select_filter():
    img = jnp.arange(16).reshape(1, 4, 4, 1).astype(jnp.float32)
    kernel = jnp.array([1., 0]).reshape(1, 1, 2)
    smooth = depthwise_separable_convolution(img, kernel, stride=2)
    assert smooth.shape == (1, 2, 2, 1)
    assert jnp.all(smooth == jnp.array([[
        [[0.], [2.]], [[8.], [10.]]
    ]]))
