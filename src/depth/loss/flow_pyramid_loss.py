from dataclasses import dataclass
from importlib import resources
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import register_dataclass

from depth.images.load import load_frame_from_path
from depth.images.pyramid import build_image_pyramid
from depth.images.separable_convolution import conv_output_size
from depth.loss.flow_loss import patch_flow_loss_grid
from depth.model.pyramid_flow import PyramidFlowEstimator, PyramidFlowEstimation, \
    PyramidFlowEstimationParams
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimator
from depth.model.upsample import FlowUpsampler



def calc_flow_pyramid_loss(flow: PyramidFlowEstimation) -> Sequence[jax.Array]:
    losses = []
    for level in flow.pyramid:
        loss = patch_flow_loss_grid(level)
        losses.append(loss)
    return losses


def check_prior_shape(coarsest_grained_frame: jax.Array, prior: jax.Array, patch_size: int,
                      stride: int) -> None:
    B, H, W, C = coarsest_grained_frame.shape
    expected_prior_shape = (
        B,
        conv_output_size(H, patch_size, stride),
        conv_output_size(W, patch_size, stride),
        2
    )
    assert prior.shape == expected_prior_shape


def test_pyramid_loss():
    frame1_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame2_path = resources.files('depth.test_fixtures') / "frame2.png"
    frame1 = load_frame_from_path(str(frame1_path), 158)
    frame2 = load_frame_from_path(str(frame2_path), 158)
    batch1 = jnp.stack([frame1, frame2], axis=0)
    batch2 = jnp.stack([frame2, frame1], axis=0)
    rngs = nnx.Rngs(0)
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=1, train=False, rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)
    upsampler = FlowUpsampler(rngs=rngs)
    pyramid_flow_estimator = PyramidFlowEstimator(
        level_flow_estimator=level_flow_estimator, upsampler=upsampler
    )
    pyramid1 = build_image_pyramid(batch1, levels=5, keep=5)
    pyramid2 = build_image_pyramid(batch2, levels=5, keep=5)
    prior = jnp.zeros((2, 3, 3, 2), jnp.float32)
    params = PyramidFlowEstimationParams(pyramid1=pyramid1, pyramid2=pyramid2, prior=prior)
    flow = pyramid_flow_estimator(params)
    losses = calc_flow_pyramid_loss(flow)
    assert losses[-1].shape == (2, 3, 3)
    assert losses[-2].shape == (2, 8, 8)
    assert losses[-3].shape == (2, 18, 18)
    assert losses[-4].shape == (2, 38, 38)
    assert losses[-5].shape == (2, 78, 78)
