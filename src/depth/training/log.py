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
    fig, axs = plt.subplots(rows, 6, figsize=(12, 2 * rows))
    column_titles = ['Frame1', 'Reflowed-F2->F1', 'Frame2', 'loss', 'flow-y', 'flow-x']
    if rows == 1:
        axs = [axs] #pyplot doesn't return a list when the there's a single row/col.
    for i, ax in enumerate(axs[0]):
        ax.set_title(column_titles[i], fontsize=14, pad=10)  # Set title for top subplot in column

    for i, (img1, img2, flow, ax) in enumerate(zip(pyramid1, pyramid2, flow_with_loss, axs)):
        ax[0].imshow(img1[0], cmap="grey", vmin=0, vmax=1)
        reflowed_img2 = apply_flow_entire_image(img2[0], flow[0, :, :, 0:2])
        ax[1].imshow(reflowed_img2, cmap="grey", vmin=0, vmax=1)
        ax[2].imshow(img2[0], cmap="grey", vmin=0, vmax=1)
        ax[3].imshow(flow[0, :, :, 2:3], cmap="grey", vmin=0, vmax=1)
        flow_max = max_expected_flow(rows - i - 1)
        ax[4].imshow(flow[0, :, :, 1:2], cmap="viridis", vmin=-flow_max, vmax=flow_max)
        ax[5].imshow(flow[0, :, :, 0:1], cmap="viridis", vmin=-flow_max, vmax=flow_max)
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
