from importlib import resources
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from tensorboardX import SummaryWriter

from depth.images.upscale import upscale_size_2n_plus_2
from depth.images.load import load_frame_from_path
from depth.model.build import make_model, generate_zero_priors


def img_post_process(img: jax.Array, scale: bool = True, normalize: bool = False):
    """tensorboard expects images as np.uint8 and in CHW dimensions.
    This takes an image with floating point values from 0 to 1 and in HWC dimensions and
    converts it.
    """
    if normalize:
        img = (img - jnp.min(img)) / (1e-6 + jnp.max(img) - jnp.min(img))
    if scale:
        img = (img * 255).astype(jnp.uint8)
    return img


def apply_flow_entire_image(img: jax.Array, flow: jax.Array) -> jax.Array:
    flow = upscale_size_2n_plus_2(flow[None, :, :, :])[0]
    flow_y = flow[:, :, 0]
    flow_x = flow[:, :, 1]
    H, W = flow_y.shape
    grid_y, grid_x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    return map_coordinates(img.squeeze(-1), [grid_y + flow_y, grid_x + flow_x], order=1)


def build_image_grid(pyramid1: Sequence[jax.Array],
                     pyramid2: Sequence[jax.Array],
                     flow_with_loss: Sequence[jax.Array]) -> Sequence[Sequence[jax.Array]]:
    grid = []
    for img1, img2, flow in zip(pyramid1, pyramid2, flow_with_loss):
        reflowed_img2 = apply_flow_entire_image(img2[0], flow[0, :, :, 0:2])[:, :, None]
        img1_out = img_post_process(img1[0], scale=False, normalize=True)
        img2_out = img_post_process(img2[0], scale=False, normalize=True)
        flow_y_out = img_post_process(flow[0, :, :, 0:1], scale=True, normalize=True)
        flow_x_out = img_post_process(flow[0, :, :, 1:2], scale=True, normalize=True)
        reflowed_img2_out = img_post_process(reflowed_img2, scale=False, normalize=False)
        loss_out = img_post_process(flow[0, :, :, 2:3], scale=True, normalize=True)
        row = [img1_out, reflowed_img2_out, img2_out, loss_out, flow_y_out, flow_x_out]
        grid.append(row)
    return grid


def log_flow_grid(pyramid1: Sequence[jax.Array],
                  pyramid2: Sequence[jax.Array],
                  flow_with_loss: Sequence[jax.Array], writer: SummaryWriter, step: int):
    grid = build_image_grid(pyramid1, pyramid2, flow_with_loss)
    img_labels = ['frame1', 'reflowed-frame2', 'frame2', 'reflowed-loss', 'flow-y', 'flow-x']
    for i, row in enumerate(grid):
        for label, img in zip(img_labels, row):
            writer.add_image(f"{i}/{label}/", np.array(img), global_step=step, dataformats='HWC')


def plot_inference_grid():
    from depth.model.train import MODEL_SETTINGS as settings
    import matplotlib.pyplot as plt

    model = make_model(0, train=False, settings=settings)
    frame1_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame2_path = resources.files('depth.test_fixtures') / "frame2.png"
    frame1 = load_frame_from_path(str(frame1_path), settings.img_size)
    frame2 = load_frame_from_path(str(frame2_path), settings.img_size)
    priors = generate_zero_priors(1, settings)

    flow_with_loss_pyramid, p1, p2 = model(frame1[None, :, :, :], frame2[None, :, :, :], priors)

    grid = build_image_grid(p1, p2, flow_with_loss_pyramid)
    fig, axs = plt.subplots(len(grid), len(grid[0]))
    for row, row_axs in zip(grid, axs):
        for img, ax in zip(row, row_axs):
            ax.imshow(img)
    plt.show()


if __name__ == '__main__':
    plot_inference_grid()
