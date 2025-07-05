from dataclasses import dataclass

import jax
from flax import nnx
from jax.tree_util import register_dataclass
import jax.numpy as jnp

from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimation, LevelFlowEstimationParams
from depth.model.single_level_flow import LevelFlowEstimator
from depth.model.upsample import FlowUpsampler
from depth.patches.loss import patch_flow_loss
from depth.images.flow import apply_flow_multi_channel_image


def patch_match_score_loss(score: jax.Array, patch_loss: jax.Array):
    return jnp.abs(score - jnp.mean(patch_loss, axis=-1))


def upsampled_flow_loss_single_batch(frame1: jax.Array, frame2: jax.Array, flow: jax.Array,
                                     confidence: jax.Array) -> jax.Array:
    mean_confidence = jnp.mean(confidence) + 1e-6
    normalized_confidence = confidence / mean_confidence
    reflowed_frame2 = apply_flow_multi_channel_image(frame2, flow)
    return jnp.abs(frame1 - reflowed_frame2) * normalized_confidence


upsampled_flow_loss = jax.vmap(upsampled_flow_loss_single_batch, in_axes=(0, 0, 0, 0))


@register_dataclass
@dataclass
class LevelFlowLossTrace:
    patch_loss: jax.Array
    upsampled_loss: jax.Array
    match_score_loss: jax.Array

    def check_shapes_consistent(self):
        B, H, W, C = self.patch_loss.shape
        assert self.match_score_loss.shape == (B, H, W, 1)
        assert self.upsampled_loss.shape == (B, 2 * H + 2, 2 * W + 2, C)


@register_dataclass
@dataclass
class LevelLossSummary:
    patch_loss: jax.Array
    upsampled_loss: jax.Array
    match_score_loss: jax.Array
    loss: jax.Array
    trace: LevelFlowLossTrace

    def check_shapes_consistent(self):
        self.trace.check_shapes_consistent()

    @classmethod
    def from_trace(cls, trace: LevelFlowLossTrace):
        B, H, W, C = trace.upsampled_loss.shape
        PB, PH, PW, C = trace.patch_loss.shape
        patch_loss = jnp.sum(trace.patch_loss) / (B * PH * PW * C)
        upsampled_loss = jnp.sum(trace.upsampled_loss) / (B * H * W * C)
        match_score_loss = jnp.sum(trace.match_score_loss) / (B * PH * PW * C)
        return LevelLossSummary(
            patch_loss=patch_loss,
            upsampled_loss=upsampled_loss,
            match_score_loss=match_score_loss,
            loss=1 * patch_loss + 1 * upsampled_loss + 0.01 * match_score_loss,
            trace=trace,
        )


def level_flow_loss(flow: LevelFlowEstimation) -> LevelLossSummary:
    """Calculates the patch loss over a grid of patches.

    Mostly a convenience function for processing patches of an image while keeping the patch
    spatial relationship in the parameters and return shapes.
    """
    flow.check_shapes_consistent()
    B, PY, PX, PH, PW, C = flow.patches1.shape
    flat_patches1 = flow.patches1.reshape(-1, PH, PW, C)
    flat_patches2 = flow.patches2.reshape(-1, PH, PW, C)
    flat_flow = flow.net_flow.reshape(-1, 2)
    flat_losses = jax.vmap(patch_flow_loss)(flat_patches1, flat_patches2, flat_flow)
    match_scores_flat = flow.match_score.reshape(-1)
    match_score_loss = patch_match_score_loss(match_scores_flat, jax.lax.stop_gradient(flat_losses))
    upsampled_loss = upsampled_flow_loss(
        flow.frame1, flow.frame2,
        flow.upsampled_flow.upsampled_flow,
        flow.upsampled_flow.fwd_confidence
    )
    level_trace = LevelFlowLossTrace(
        patch_loss=flat_losses.reshape(B, PY, PX, C),
        upsampled_loss=upsampled_loss.reshape(B, 2 * PY + 2, 2 * PX + 2, C),  # TODO: PARAMETRIZE
        match_score_loss=match_score_loss.reshape(B, PY, PX, 1),
    )
    summary = LevelLossSummary.from_trace(level_trace)
    return summary


def test_single_level_flow():
    rngs = nnx.Rngs(0)
    img = jax.random.uniform(jax.random.key(1), (3, 6, 8, 2))
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=2, train=False, rngs=rngs
    )
    upsampler = FlowUpsampler(rngs=rngs)
    level_flow_estimator = LevelFlowEstimator(
        stride=2,
        flow_estimator=patch_flow_estimator,
        upsampler=upsampler
    )
    prior = jax.random.uniform(jax.random.key(2), (3, 2, 3, 2))
    confidence = jax.random.uniform(jax.random.key(3), (3, 2, 3, 1))
    params = LevelFlowEstimationParams(frame1=img, frame2=img, prior=prior, confidence=confidence)
    flow = level_flow_estimator(params)
    loss = level_flow_loss(flow)
    loss.check_shapes_consistent()
