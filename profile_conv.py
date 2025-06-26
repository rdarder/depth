# profiling_separable_convolutions.py
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import pandas as pd
import \
    numpy as np  # Using numpy for general array operations before converting to JAX arrays if needed


# --- Provided Convolution Implementations ---

def conv_output_size(input_size: int, kernel_size: int, stride: int) -> int:
    """Calculates the output dimension for a convolution with 'VALID' padding."""
    return (input_size - kernel_size) // stride + 1


def depthwise_separable_convolution_repeat(imgs: jax.Array, kernel: jax.Array, stride: int):
    N, H, W, C = imgs.shape

    # The input 'kernel' is a 1D kernel (e.g., shape (1, 1, K)).
    # We flatten it to get the 1D coefficients.
    kernel_1d = kernel.flatten()

    # Create the 2D separable kernel using the outer product.
    # This results in a (K, K) kernel where K[i, j] = kernel_1d[i] * kernel_1d[j].
    kernel_2d_base = jnp.outer(kernel_1d, kernel_1d)

    # Reshape the 2D kernel for jax.lax.conv_general_dilated in 'HWIO' format.
    # Since this is a per-channel convolution (depthwise-like),
    # the input and output channels for the filter itself are 1.
    # The full kernel shape will be (kernel_height, kernel_width, input_channels, output_channels).
    kernel_for_conv = kernel_2d_base[:, :, None, None].repeat(C, axis=3)
    # Perform the single 2D convolution.
    # We use 'feature_group_count=C' to apply the same filter to each input channel independently,
    # effectively performing C parallel 2D convolutions.
    output = jax.lax.conv_general_dilated(
        imgs,
        kernel_for_conv,
        window_strides=(stride, stride),  # 2D strides
        padding='VALID',  # No padding, matching original
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),  # Standard NHWC input/output, HWIO kernel
        feature_group_count=C  # Apply convolution per input channel
    )

    return output


def depthwise_separable_convolution_no_channel(imgs: jax.Array, kernel: jax.Array, stride: int):
    N, H, W = imgs.shape

    # The input 'kernel' is a 1D kernel (e.g., shape (1, 1, K)).
    # We flatten it to get the 1D coefficients.
    kernel_1d = kernel.flatten()

    # Create the 2D separable kernel using the outer product.
    # This results in a (K, K) kernel where K[i, j] = kernel_1d[i] * kernel_1d[j].
    kernel_2d_base = jnp.outer(kernel_1d, kernel_1d)

    # Reshape the 2D kernel for jax.lax.conv_general_dilated in 'HWIO' format.
    # Since this is a per-channel convolution (depthwise-like),
    # the input and output channels for the filter itself are 1.
    # The full kernel shape will be (kernel_height, kernel_width, input_channels, output_channels).
    kernel_for_conv = kernel_2d_base[:, :, None, None].repeat(C, axis=3)
    # Perform the single 2D convolution.
    # We use 'feature_group_count=C' to apply the same filter to each input channel independently,
    # effectively performing C parallel 2D convolutions.
    output = jax.lax.conv_general_dilated(
        imgs[:, :, :, None],
        kernel_for_conv,
        window_strides=(stride, stride),  # 2D strides
        padding='VALID',  # No padding, matching original
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),  # Standard NHWC input/output, HWIO kernel
    )

    return output


depthwise_separable_convolution_vmap = jax.vmap(depthwise_separable_convolution_no_channel,
                                                in_axes=(3, None, None))


def depthwise_separable_convolution_1d(imgs: jax.Array, kernel: jax.Array, stride: int):
    N, H, W, C = imgs.shape
    kernel_size = kernel.shape[-1]
    OH = conv_output_size(H, kernel_size, stride=stride)
    OW = conv_output_size(W, kernel_size, stride=stride)

    # First 1D convolution (horizontal pass)
    # Reshape input for 1D convolution: (N*H*C, W, 1)
    # Transpose H and W to make W the convolution dimension for horizontal pass.
    img_for_h_conv = imgs.transpose(0, 1, 3, 2).reshape(N * H * C, W, 1)

    img_h_conv = jax.lax.conv_general_dilated(
        img_for_h_conv,
        kernel,  # Kernel is (1, K) or (K,) for 1D, so its last dim is K.
        window_strides=(stride,),
        padding='VALID',
        dimension_numbers=('NWC', 'IOW', 'NWC'),
        # NWC for input/output, IOW for kernel (Input_channels, Output_channels, Weight_dimension)
    )  # Output shape: (N*H*C, OW, 1)

    # Second 1D convolution (vertical pass)
    # Reshape the output of the horizontal pass for vertical convolution.
    # We need to reshape back to (N, H, C, OW), then transpose to make H the convolution dimension.
    # The new N dimension for this conv will be N * C * OW.
    img_for_v_conv = (
        img_h_conv.reshape(N, H, C, OW)
        .transpose(0, 2, 3, 1)  # Rearrange to (N, C, OW, H)
        .reshape(N * C * OW, H, 1)  # Reshape to (N', H, 1) where N' = N*C*OW
    )

    img_v_conv = jax.lax.conv_general_dilated(
        img_for_v_conv,
        kernel,  # Same 1D kernel used for the vertical pass
        window_strides=(stride,),
        padding='VALID',
        dimension_numbers=('NHC', 'IOH', 'NHC'),
        # NHC for input/output, IOH for kernel (Input_channels, Output_channels, Height_dimension)
    )  # Output shape: (N * C * OW, OH, 1)

    # Reshape the final output back to NHWC format.
    output = (
        img_v_conv.reshape(N, C, OW, OH)  # Reshape to (N, C, OW, OH)
        .transpose(0, 3, 2, 1)  # Transpose to (N, OH, OW, C) to match NHWC
    )
    return output


def depthwise_separable_convolution_1d_no_channel(imgs: jax.Array, kernel: jax.Array, stride: int):
    N, H, W = imgs.shape
    kernel_size = kernel.shape[-1]
    OH = conv_output_size(H, kernel_size, stride=stride)
    OW = conv_output_size(W, kernel_size, stride=stride)

    # First 1D convolution (horizontal pass)
    # Reshape input for 1D convolution: (N*H*C, W, 1)
    # Transpose H and W to make W the convolution dimension for horizontal pass.
    img_for_h_conv = imgs.reshape(N * H, W, 1)

    img_h_conv = jax.lax.conv_general_dilated(
        img_for_h_conv,
        kernel,  # Kernel is (1, K) or (K,) for 1D, so its last dim is K.
        window_strides=(stride,),
        padding='VALID',
        dimension_numbers=('NWC', 'IOW', 'NWC'),
        # NWC for input/output, IOW for kernel (Input_channels, Output_channels, Weight_dimension)
    )  # Output shape: (N*H*C, OW, 1)

    # Second 1D convolution (vertical pass)
    # Reshape the output of the horizontal pass for vertical convolution.
    # We need to reshape back to (N, H, C, OW), then transpose to make H the convolution dimension.
    # The new N dimension for this conv will be N * C * OW.
    img_for_v_conv = (
        img_h_conv.reshape(N, H, OW)
        .transpose(0, 2, 1)
        .reshape(N * OW, H, 1)  # Reshape to (N', H, 1) where N' = N*C*OW
    )

    img_v_conv = jax.lax.conv_general_dilated(
        img_for_v_conv,
        kernel,  # Same 1D kernel used for the vertical pass
        window_strides=(stride,),
        padding='VALID',
        dimension_numbers=('NHC', 'IOH', 'NHC'),
        # NHC for input/output, IOH for kernel (Input_channels, Output_channels, Height_dimension)
    )  # Output shape: (N * C * OW, OH, 1)

    # Reshape the final output back to NHWC format.
    output = (
        img_v_conv.reshape(N, OW, OH, 1)  # Reshape to (N, C, OW, OH)
        .transpose(0, 2, 1, 3)  # Transpose to (N, OH, OW, C) to match NHWC
    )
    return output


depthwise_separable_convolution_1d_no_channel_vmap = jax.vmap(
    depthwise_separable_convolution_1d_no_channel, in_axes=(3, None, None))
# --- Profiling Setup ---

# JIT compile the functions for performance
jitted_depthwise_separable_convolution = jax.jit(depthwise_separable_convolution_repeat,
                                                 static_argnums=2)
jitted_depthwise_separable_convolution_vmap = jax.jit(depthwise_separable_convolution_vmap,
                                                      static_argnums=2)
jitted_depthwise_separable_convolution_1d = jax.jit(depthwise_separable_convolution_1d,
                                                    static_argnums=2)
jitted_depthwise_separable_convolution_1d_vmap = jax.jit(
    depthwise_separable_convolution_1d_no_channel_vmap,
    static_argnums=2)

to_profile = {
    '2d_repeat': jitted_depthwise_separable_convolution,
    '2d_vmap': jitted_depthwise_separable_convolution_vmap,
    '1d': jitted_depthwise_separable_convolution_1d,
    '1d_vmap': jitted_depthwise_separable_convolution_1d_vmap
}

# Define profiling parameters
Ns = [1, 10, 100]
HWs = [32, 64, 128, 256]
Cs = [1, 2, 4]
STRIDE = 1  # Fixed stride as not varied by user
KERNEL_SIZE = 3  # Fixed kernel size, assuming a 1D kernel like (3,) or (1,3)

# Store results
results_2d = []
results_1d = []

print("Starting profiling...")

# Loop through all combinations
results = {fname: [] for fname in to_profile.keys()}
for N in Ns:
    for HW in HWs:
        H, W = HW, HW
        for C in Cs:
            # Generate dummy data
            # Use jnp.ones for simplicity, as data content doesn't affect computation time significantly
            # compared to shape and operation type.
            imgs = jnp.ones((N, H, W, C), dtype=jnp.float32)

            # Kernel needs to be 1D for depthwise_separable_convolution_1d (shape (1, 1, KERNEL_SIZE))
            kernel_1d_conv = jnp.ones((1, 1, KERNEL_SIZE), dtype=jnp.float32)

            # warmup
            for fname, fn in to_profile.items():
                _ = fn(imgs, kernel_1d_conv, STRIDE).block_until_ready()


            for fname, fn in to_profile.items():
                start_time = time.perf_counter()
                _ = fn(imgs, kernel_1d_conv, STRIDE).block_until_ready()
                end_time = time.perf_counter()
                time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                results[fname].append({'N': N, 'H': H, 'W': W, 'C': C, 'time_ms': time_ms})
                print(f"{fname}: N={N}, H={H}, W={W}, C={C}: {time_ms}")


print(results)
print("\nProfiling complete. Generating plots...")

# --- Plotting Results (Rewritten without defensive code, assuming pandas) ---

# Convert lists of dicts to pandas DataFrames
df_results = {key: pd.DataFrame(val) for key, val in results.items()}


# Helper function to get data for plotting using pandas
def get_plot_data_pd(df, fixed_N, fixed_HW, fixed_C, vary_param):
    """
    Filters a DataFrame for plotting.

    Args:
        df (pandas.DataFrame): The DataFrame containing profiling results.
        fixed_N (int): The N value to fix for the plot.
        fixed_HW (int): The H/W value to fix for the plot.
        fixed_C (int): The C value to fix for the plot.
        vary_param (str): The parameter to vary ('N', 'H', or 'C').

    Returns:
        tuple: (x_values, y_values) for plotting.
    """
    if vary_param == 'N':
        filtered_df = df[(df['H'] == fixed_HW) & (df['C'] == fixed_C)]
        x_values = filtered_df['N'].values
    elif vary_param == 'H':
        filtered_df = df[(df['N'] == fixed_N) & (df['C'] == fixed_C)]
        x_values = filtered_df['H'].values
    elif vary_param == 'C':
        filtered_df = df[(df['N'] == fixed_N) & (df['H'] == fixed_HW)]
        x_values = filtered_df['C'].values

    y_values = filtered_df['time_ms'].values
    return x_values, y_values


# Plotting parameters for fixed values
fixed_N_plot = 10
fixed_HW_plot = 64
fixed_C_plot = 2

# Labels and markers for each convolution type
plot_configs = {
    '2d_repeat': {'label': '2D Grouped Conv (Repeat Kernel)', 'marker': 'o'},
    '2d_vmap': {'label': '2D Conv (vmap per Channel)', 'marker': 's'},
    '1d': {'label': 'Two 1D Convs (Reshape/Transpose)', 'marker': 'x'},
    '1d_vmap': {'label': 'Two 1D Convs (vmap per Channel)', 'marker': '^'}
}

plt.figure(figsize=(20, 6))  # Increased figure size for better readability

# --- Plot 1: Performance vs. N (Batch Size) ---
plt.subplot(1, 3, 1)
for func_name, config in plot_configs.items():
    x_values, y_values = get_plot_data_pd(df_results[func_name],
                                          fixed_N=fixed_N_plot,
                                          fixed_HW=fixed_HW_plot,
                                          fixed_C=fixed_C_plot,
                                          vary_param='N')
    plt.plot(x_values, y_values, marker=config['marker'], label=config['label'])

plt.title(f'Performance vs. Batch Size (N)\n(H={fixed_HW_plot}, C={fixed_C_plot})')
plt.xlabel('Batch Size (N)')
plt.ylabel('Time (ms)')
plt.xscale('log')  # Log scale for N
plt.yscale('log')  # Log scale for time
plt.grid(True, which="both", ls="-")
plt.legend()

# --- Plot 2: Performance vs. H=W (Image Dimension) ---
plt.subplot(1, 3, 2)
for func_name, config in plot_configs.items():
    x_values, y_values = get_plot_data_pd(df_results[func_name],
                                          fixed_N=fixed_N_plot,
                                          fixed_HW=fixed_HW_plot,
                                          fixed_C=fixed_C_plot,
                                          vary_param='H')
    plt.plot(x_values, y_values, marker=config['marker'], label=config['label'])

plt.title(f'Performance vs. Image Dimension (H=W)\n(N={fixed_N_plot}, C={fixed_C_plot})')
plt.xlabel('Image Dimension (H=W)')
plt.ylabel('Time (ms)')
plt.xscale('log')  # Log scale for H=W
plt.yscale('log')  # Log scale for time
plt.grid(True, which="both", ls="-")
plt.legend()

# --- Plot 3: Performance vs. C (Channels) ---
plt.subplot(1, 3, 3)
for func_name, config in plot_configs.items():
    x_values, y_values = get_plot_data_pd(df_results[func_name],
                                          fixed_N=fixed_N_plot,
                                          fixed_HW=fixed_HW_plot,
                                          fixed_C=fixed_C_plot,
                                          vary_param='C')
    plt.plot(x_values, y_values, marker=config['marker'], label=config['label'])

plt.title(f'Performance vs. Channels (C)\n(N={fixed_N_plot}, H={fixed_HW_plot})')
plt.xlabel('Channels (C)')
plt.ylabel('Time (ms)')
plt.xscale('log')  # Log scale for C
plt.yscale('log')  # Log scale for time
plt.grid(True, which="both", ls="-")
plt.legend()

plt.tight_layout()  # Adjust subplot params for a tight layout
plt.show()