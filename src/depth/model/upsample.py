from dataclasses import dataclass, field
from functools import partial

import einops
import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import register_dataclass

from depth.jax_utils import print_shapes


@register_dataclass
@dataclass
class FlowUpsamplerParams:
    residual_flow: jax.Array
    net_flow: jax.Array
    confidence: jax.Array
    match_score: jax.Array
    patch_score: jax.Array

    def check_consistent_shapes(self):
        assert self.residual_flow.shape == self.net_flow.shape
        B, H, W, F = self.net_flow.shape
        assert self.confidence.shape == (B, H, W, 1)
        assert self.match_score.shape == (B, H, W, 1)
        assert self.patch_score.shape == (B, H, W, 1)


@register_dataclass
@dataclass
class UpsampledFlow:
    upsampled_flow: jax.Array
    fwd_confidence: jax.Array

    def check_shapes_consistent_with_params(self, params: FlowUpsamplerParams):
        params.check_consistent_shapes()
        B, H, W, F = params.net_flow.shape
        assert self.upsampled_flow.shape == (B, 2 * H + 2, 2 * W + 2, F)
        assert self.fwd_confidence.shape == (B, 2 * H + 2, 2 * W + 2, 1)


class FlowUpsampler(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.influence_receptiveness_conv = nnx.Conv(
            in_features=5,
            out_features=2,
            kernel_size=(1, 1),
            strides=(1, 1),
            rngs=rngs
        )
        self.spatial_influence_conv = nnx.Conv(
            in_features=1,
            out_features=4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1, use_bias=False,
            kernel_init=nnx.initializers.ones,
            rngs=rngs)

    def __call__(self, params: FlowUpsamplerParams) -> UpsampledFlow:
        params.check_consistent_shapes()
        influence_input = jnp.concatenate(
            [params.residual_flow, params.patch_score, params.match_score, params.confidence],
            axis=-1
        )
        influence_receptiveness_grid = jax.nn.sigmoid(self.influence_receptiveness_conv(
            influence_input))
        influence_grid, receptiveness_grid = jnp.split(influence_receptiveness_grid, 2, axis=-1)
        outgoing_flows = jnp.einsum('bhw1,bhwf->bhwf', influence_grid, params.net_flow)
        spatial_influence_norms = self.spatial_influence_conv(influence_grid)
        incoming_flows = (
            jax.vmap(self.spatial_influence_conv, in_axes=(3,), out_axes=4)
            (outgoing_flows[:, :, :, :, None])
        )

        influence_flow = incoming_flows / (spatial_influence_norms[:, :, :, :, None] + 1e-8)
        receptiveness_grid_broadcast = receptiveness_grid[:, :, :, :, None]
        retained_flow = ((1 - receptiveness_grid) * params.net_flow)[:, :, :, None, :]
        effective_flow = receptiveness_grid_broadcast * influence_flow + retained_flow
        upsampled_flow = einops.rearrange(
            effective_flow,  # B H W (00 01 10 11) (yx)
            'b h w (p_h p_w) f -> b (h p_h) (w p_w) f', p_h=2, p_w=2
        )
        padded_upsampled_flow = jnp.pad(
            upsampled_flow,
            pad_width=((0, 0), (1, 1), (1, 1), (0, 0)),
            mode='edge'
        )
        forward_confidence = einops.rearrange(
            spatial_influence_norms,  # B H W (00 01 10 11) (yx)
            'b h w (p_h p_w)-> b (h p_h) (w p_w) 1', p_h=2, p_w=2

        )
        padded_forward_confidence = jnp.pad(
            forward_confidence,
            pad_width=((0, 0), (1, 1), (1, 1), (0, 0)),
            mode='constant'
        )
        upsampled = UpsampledFlow(
            upsampled_flow=padded_upsampled_flow,
            fwd_confidence=padded_forward_confidence
        )
        upsampled.check_shapes_consistent_with_params(params)
        return upsampled


def test_upscale_flow():
    rngs = nnx.Rngs(0)
    upsampler = FlowUpsampler(rngs=rngs)
    params = FlowUpsamplerParams(
        net_flow=jax.random.normal(rngs.flow(), (3, 8, 7, 2)) * 2,
        residual_flow=jax.random.normal(rngs.flow(), (3, 8, 7, 2)),
        patch_score=jax.random.normal(rngs.patch(), (3, 8, 7, 1)),
        match_score=jax.random.normal(rngs.match(), (3, 8, 7, 1)),
        confidence=jax.random.normal(rngs.fwd_confidence(), (3, 8, 7, 1)),
    )
    upsampled: UpsampledFlow = upsampler(params)
    upsampled.check_shapes_consistent_with_params(params)
    assert upsampled.upsampled_flow.shape == (3, 18, 16, 2)
    assert upsampled.fwd_confidence.shape == (3, 18, 16, 1)
