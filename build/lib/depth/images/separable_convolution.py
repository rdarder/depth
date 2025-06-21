import jax


def separable_convolution(imgs: jax.Array, kernel: jax.Array, stride: int):
    N, C, H, W = imgs.shape
    kernel_size = kernel.shape[-1]
    OH = conv_output_size(H, kernel_size, stride=stride)
    OW = conv_output_size(W, kernel_size, stride=stride)
    img_for_h_conv = imgs.reshape(N * C * H, 1, W)
    img_h_conv = jax.lax.conv_general_dilated(
        img_for_h_conv,
        kernel,
        window_strides=(stride,),
        padding=((0, 0),),
        dimension_numbers=('NCW', 'IOW', 'NCW'),
    ).reshape(N, C, H, OW)
    img_for_v_conv = img_h_conv.transpose(0, 1, 3, 2).reshape(N * C * OW, 1, H)
    img_v_conv = jax.lax.conv_general_dilated(
        img_for_v_conv,
        kernel,
        window_strides=(stride,),
        padding=((0, 0),),
        dimension_numbers=('NCH', 'IOH', 'NCH'),
    ).reshape(N, C, OW, OH)
    return img_v_conv.transpose(0, 1, 3, 2)


def conv_output_size(input_size: int, kernel_size: int, stride: int):
    return (input_size - kernel_size) // stride + 1
