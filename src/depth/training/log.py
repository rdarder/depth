from importlib import resources
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.ndimage import map_coordinates
from matplotlib.figure import Figure
from tensorboardX import SummaryWriter

from depth.images.load import load_frame_from_path
from depth.images.upscale import upscale_size_2n_plus_2
from depth.model.build import make_model
from depth.model.patch_flow import PatchFlowEstimator
from depth.train.build import generate_zero_priors
from depth.model.settings import ModelSettings


def apply_flow_entire_image(img: jax.Array, flow: jax.Array) -> jax.Array:
    flow = upscale_size_2n_plus_2(flow[None, :, :, :])[0]
    flow_y = flow[:, :, 0]
    flow_x = flow[:, :, 1]
    H, W = flow_y.shape
    grid_y, grid_x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    return map_coordinates(img.squeeze(-1), [grid_y + flow_y, grid_x + flow_x], order=1)


def max_expected_flow(inv_level: int):
    if inv_level == 0:
        return 0.5
    else:
        return 0.5 + 2 * max_expected_flow(inv_level - 1)


def build_image_grid(pyramid1: Sequence[jax.Array],
                     pyramid2: Sequence[jax.Array],
                     flow_with_loss: Sequence[jax.Array]) -> Figure:
    rows = len(pyramid1)
    cols = 8
    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    column_titles = ['Frame1', 'Reflowed-F2->F1', 'Frame2', 'frame-diff', 'reflow-diff',
                     'loss', 'flow-y', 'flow-x']
    if rows == 1:
        axs = [axs]  # pyplot doesn't return a list when the there's a single row/col.
    for i, ax in enumerate(axs[0]):
        ax.set_title(column_titles[i], fontsize=14, pad=10)  # Set title for top subplot in column

    for i, (img1, img2, flow, ax) in enumerate(zip(pyramid1, pyramid2, flow_with_loss, axs)):
        ax[0].imshow(img1[0], cmap="grey", vmin=0, vmax=1)
        reflowed_img2 = apply_flow_entire_image(img2[0], flow[0, :, :, 0:2])
        ax[1].imshow(reflowed_img2, cmap="grey", vmin=0, vmax=1)
        ax[2].imshow(img2[0], cmap="grey", vmin=0, vmax=1)
        ax[3].imshow(img2[0] - img1[0], cmap="coolwarm", vmin=-1, vmax=1)
        ax[4].imshow(reflowed_img2[:, :, None] - img1[0], cmap="coolwarm", vmin=-1, vmax=1)
        ax[5].imshow(flow[0, :, :, 2:3], cmap="grey", vmin=0, vmax=1)
        flow_max = max_expected_flow(rows - i - 1)
        ax[6].imshow(flow[0, :, :, 1:2], cmap="coolwarm", vmin=-flow_max, vmax=flow_max)
        ax[7].imshow(flow[0, :, :, 0:1], cmap="coolwarm", vmin=-flow_max, vmax=flow_max)
        for axc in ax:
            axc.set_axis_off()
    plt.tight_layout()
    return fig


def log_flow_grid(pyramid1: Sequence[jax.Array],
                  pyramid2: Sequence[jax.Array],
                  flow_with_loss: Sequence[jax.Array], writer: SummaryWriter, step: int):
    fig = build_image_grid(pyramid1, pyramid2, flow_with_loss)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    writer.add_image("sample_inference", img, step, dataformats='HWC')


def plot_inference_grid():
    settings = ModelSettings()
    model = make_model(0, train=False, settings=settings)
    frame1_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame2_path = resources.files('depth.test_fixtures') / "frame2.png"
    frame1 = load_frame_from_path(str(frame1_path), settings.img_size)
    frame2 = load_frame_from_path(str(frame2_path), settings.img_size)
    priors = generate_zero_priors(1, settings)

    flow_with_loss_pyramid, p1, p2 = model(frame1[None, :, :, :], frame2[None, :, :, :], priors)

    build_image_grid(p1, p2, flow_with_loss_pyramid)
    plt.show()


if __name__ == '__main__':
    plot_inference_grid()


def log_train_progress(aux, global_step, loss_value, writer):
    coarse_to_fine_losses = reversed(aux['levels_losses'])
    for i, level_loss in enumerate(coarse_to_fine_losses):
        writer.add_scalar(f"level_loss/{i}", level_loss, global_step)
    writer.add_scalar("train_loss", loss_value, global_step)
    log_flow_grid(aux['pyramid1'], aux['pyramid2'], aux['flow_with_loss'], writer,
                  global_step)
    print(
        f"Step {global_step:06}\n"
        f"    Total Weighted Loss: {loss_value:.4f}\n"
        f"    Levels losses: {aux['levels_losses']}\n"
        f"    Levels weights: {aux['levels_weights']}\n"
    )


def visualize_shift_conv_weights(model: PatchFlowEstimator) -> np.ndarray:
    """
    Generates an image from the shift_conv kernel weights for visualization.

    The function extracts the 2x2 filters, normalizes each filter's weights
    to the [0, 1] range, and arranges them into a grid.

    Args:
        model: An instance of the PatchFlowEstimator model.

    Returns:
        A single numpy array representing the grid of filter weights in the
        format (H, W, 1), suitable for logging as an image to TensorBoard.
    """
    # The kernel is a depthwise convolution, so its shape is
    # (2, 2, 1, 8 * num_channels)
    weights = model.shift_conv.kernel

    # Squeeze the grouped input channel dim, shape becomes (2, 2, 8 * num_channels)
    weights = jnp.squeeze(weights, axis=2)

    # Transpose to (num_filters, H, W) -> (8 * num_channels, 2, 2)
    weights = jnp.transpose(weights, (2, 0, 1))

    # Normalize each 2x2 filter individually to the [0, 1] range
    min_vals = jnp.min(weights, axis=(1, 2), keepdims=True)
    max_vals = jnp.max(weights, axis=(1, 2), keepdims=True)
    # Add a small epsilon to avoid division by zero for constant-valued filters
    normalized_weights = (weights - min_vals) / (max_vals - min_vals + 1e-6)

    # We have 8 * num_channels filters. Let's arrange them in a grid.
    # A grid with 4 rows is a reasonable choice.
    num_filters = normalized_weights.shape[0]
    grid_rows = 4
    grid_cols = num_filters // grid_rows

    # Reshape into a grid of filters: (grid_rows, grid_cols, 2, 2)
    grid = normalized_weights.reshape(grid_rows, grid_cols, 2, 2)

    # Transpose and reshape to form a single image:
    # (grid_rows, 2, grid_cols, 2) -> (grid_rows * 2, grid_cols * 2)
    grid = grid.transpose(0, 2, 1, 3)
    image = grid.reshape(grid_rows * 2, grid_cols * 2)

    # Add a channel dimension for image logging (H, W, C)
    image_with_channel = jnp.expand_dims(image, axis=-1)

    # Return as a NumPy array for compatibility with I/O and logging libraries
    return np.asarray(image_with_channel)
