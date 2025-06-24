import jax
import jax.numpy as jnp


def separable_convolution(imgs: jax.Array, kernel: jax.Array, stride: int):
    N, H, W, C = imgs.shape
    kernel_size = kernel.shape[-1]
    OH = conv_output_size(H, kernel_size, stride=stride)
    OW = conv_output_size(W, kernel_size, stride=stride)
    img_for_h_conv = imgs.transpose(0, 1, 3, 2).reshape(N * H * C, W, 1)
    img_h_conv = jax.lax.conv_general_dilated(
        img_for_h_conv,
        kernel,
        window_strides=(stride,),
        padding='VALID',
        dimension_numbers=('NWC', 'IOW', 'NWC'),
    )  # (N*H*C, OW, 1)
    img_for_v_conv = (
        img_h_conv.reshape(N, H, C, OW)
        .transpose(0, 2, 3, 1)
        .reshape(N * C * OW, H, 1)
    )
    img_v_conv = jax.lax.conv_general_dilated(
        img_for_v_conv,
        kernel,
        window_strides=(stride,),
        padding='VALID',
        dimension_numbers=('NHC', 'IOH', 'NHC'),
    )  # (N * C * OW, OH, 1)
    output = (
        img_v_conv.reshape(N, C, OW, OH)
        .transpose(0, 3, 2, 1)
    )
    return output


def conv_output_size(input_size: int, kernel_size: int, stride: int):
    output = (input_size - kernel_size) // stride + 1
    assert output > 0
    return output


def conv_output_size_steps(input_size: int, kernel_size: int, stride: int, steps: int):
    output = input_size
    for i in range(steps):
        output = conv_output_size(output, kernel_size, stride)
    return output


def test_separable_smoothing_filter():
    img = jnp.arange(9).reshape(1, 3, 3, 1).astype(jnp.float32)
    kernel = jnp.array([0.5, 0.5]).reshape(1, 1, 2)
    smooth = separable_convolution(img, kernel, stride=1)
    assert smooth.shape == (1, 2, 2, 1)
    assert jnp.all(smooth == jnp.array([[
        [[2.], [3.]], [[5.], [6.]]
    ]]))


def test_separable_select_filter():
    img = jnp.arange(16).reshape(1, 4, 4, 1).astype(jnp.float32)
    kernel = jnp.array([1., 0]).reshape(1, 1, 2)
    smooth = separable_convolution(img, kernel, stride=2)
    assert smooth.shape == (1, 2, 2, 1)
    assert jnp.all(smooth == jnp.array([[
        [[0.], [2.]], [[8.], [10.]]
    ]]))
