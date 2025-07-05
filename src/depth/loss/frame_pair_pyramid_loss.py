from dataclasses import dataclass
from importlib import resources
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import register_dataclass

from depth.images.load import load_frame_from_path
from depth.images.pyramid import build_image_pyramid
from depth.loss.flow_loss import level_flow_loss, LevelLossSummary
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.pyramid_flow import PyramidFlowEstimator, PyramidFlowEstimationParams, \
    PyramidFlowEstimation
from depth.model.single_level_flow import LevelFlowEstimator
from depth.model.upsample import FlowUpsampler


@register_dataclass
@dataclass
class LossTrace:
    level_losses: Sequence[LevelLossSummary]
    flow: PyramidFlowEstimation
    weighted_level_losses: jax.Array
    weights: jax.Array
    weighted_loss: jax.Array

    def check_shapes_consistent(self, patch_size: int, stride: int):
        pass
        # TODO


def pyramid_loss(flow: PyramidFlowEstimation) -> LossTrace:
    level_losses = _pyramid_level_losses(flow)
    level_losses_summaries = jnp.array([level.loss for level in level_losses])
    weights = 1 / jnp.array([2 ** i for i in range(len(level_losses))])
    weights = weights / jnp.sum(weights)
    weighted_loss = jnp.sum(level_losses_summaries * weights)
    loss_trace = LossTrace(
        level_losses=level_losses,
        flow=flow,
        weighted_level_losses=level_losses_summaries,
        weights=weights,
        weighted_loss=weighted_loss,
    )
    return loss_trace


pyramid_loss_value_and_grad = nnx.value_and_grad(pyramid_loss, has_aux=True)


def _pyramid_level_losses(flow: PyramidFlowEstimation) -> Sequence[LevelLossSummary]:
    losses = []
    for level in flow.pyramid:
        loss = level_flow_loss(level)
        losses.append(loss)
    return losses


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
    upsampler = FlowUpsampler(rngs=rngs)
    level_flow_estimator = LevelFlowEstimator(
        stride=2,
        flow_estimator=patch_flow_estimator,
        upsampler=upsampler
    )
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator=level_flow_estimator)
    pyramid1 = build_image_pyramid(batch1, levels=5, keep=5)
    pyramid2 = build_image_pyramid(batch2, levels=5, keep=5)
    prior = jnp.zeros((2, 3, 3, 2), jnp.float32)
    confidence = jnp.zeros((2, 3, 3, 1), jnp.float32)
    params = PyramidFlowEstimationParams(
        pyramid1=pyramid1,
        pyramid2=pyramid2,
        prior=prior,
        confidence=confidence
    )
    flow = pyramid_flow_estimator(params)
    trace = pyramid_loss(flow)
    trace.check_shapes_consistent(patch_size=4, stride=2)
