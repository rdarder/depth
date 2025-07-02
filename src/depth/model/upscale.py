import einops
import jax
import jax.numpy as jnp
from flax.experimental import nnx as nnx

from depth.jax_utils import print_shapes


class FlowUpscaler(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv_q = nnx.Conv(in_features=5, out_features=8, kernel_size=(1, 1), strides=(1, 1),
                               rngs=rngs)
        self.conv_k = nnx.Conv(in_features=5, out_features=8, kernel_size=(1, 1), strides=(1, 1),
                               rngs=rngs)
        self.conv_v = nnx.Conv(in_features=5, out_features=1, kernel_size=(1, 1), strides=(1, 1),
                               rngs=rngs)
        self.grid_weights = nnx.Param(jnp.ones((4, 9), dtype=jnp.float32) * 2)
        # am hardcoding the (2x flow upscaling) as a weight initializer here.
        # definitely not right.

    def __call__(self, input_flow_with_scores: jax.Array):
        B, H, W, S = input_flow_with_scores.shape
        assert S == 5
        q = self.conv_q(input_flow_with_scores)
        q = jax.nn.relu(q)
        k = self.conv_k(input_flow_with_scores)
        k = jax.nn.relu(k)
        v = self.conv_v(input_flow_with_scores)
        v = jax.nn.relu(v)
        # print_shapes(input_flow_with_scores=input_flow_with_scores, q=q, k=k, v=v)

        pad_q = jnp.pad(q, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
        q_neighbors = []
        for dy in range(3):  # Shift for rows
            for dx in range(3):  # Shift for columns
                # Slice an (H, W) window from pad_q starting at (dy, dx)
                q_neighbor = pad_q[:, dy:dy + H, dx:dx + W, :]
                q_neighbors.append(q_neighbor)

        q_patches = jnp.stack(q_neighbors, axis=-1)
        qk = jnp.sum(q_patches * k[..., None], axis=3)
        # print_shapes(pad_q=pad_q, q_patches=q_patches, qk=qk)

        qkp = jnp.einsum('bhwc,pc->bhwpc', qk,
                         self.grid_weights)  # This is doing the right thing (tm)
        sharpened_qkp = jax.nn.softmax(qkp, axis=-1)
        qkpv = jnp.einsum('bhwpc,bhw1->bhwpc', sharpened_qkp, v)
        adjusted_flow_weights = jnp.einsum('bhwf,bhwpc->bhwpcf',
                                           input_flow_with_scores[:, :, :, :2], qkpv)

        # print_shapes(qkp=qkp, sharpened_qkp=sharpened_qkp,
        #              qkpv=qkpv, adjusted_flow_weights=adjusted_flow_weights)

        pfw = jnp.pad(adjusted_flow_weights, (
            (0, 0), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0)
        ), mode="constant")

        shifted_flow_contribution_list = []
        channel = 0
        for dy in range(2, -1, -1):  # Shift for rows
            for dx in range(2, -1, -1):  # Shift for columns
                # Slice an (H, W) window from pad_q starting at (dy, dx)
                shifted_flow_channel = pfw[:, dy:dy + H, dx:dx + W, :, channel, :]
                shifted_flow_contribution_list.append(shifted_flow_channel)
                channel += 1

        shifted_flow_contributions = jnp.stack(shifted_flow_contribution_list, axis=-2)
        # print_shapes(pfw=pfw, shifted_flow_contributions=shifted_flow_contributions)
        output_flow = jnp.einsum('bhwpcf->bhwpf', shifted_flow_contributions)

        confidence = qkp[:, :, :, :, 4]
        upscaled_confidence = einops.rearrange(confidence, 'b h w (p_h p_w) -> b (h p_h) (w p_w) 1',
                                               p_h=2,
                                               p_w=2)
        upscaled_output_flow = einops.rearrange(output_flow,
                                                'b h w (p_h p_w) f -> b (h p_h) (w p_w) f', p_h=2,
                                                p_w=2)
        # print_shapes(output_flow=output_flow,
        #              upscaled_confidence=upscaled_confidence,
        #              upscaled_output_flow=upscaled_output_flow)

        padded_output = jnp.pad(upscaled_output_flow, ((0, 0), (1, 1), (1, 1), (0, 0)),
                                mode='edge')
        padded_confidence = jnp.pad(upscaled_confidence, ((0, 0), (1, 1), (1, 1), (0, 0)),
                                    mode='constant')
        return padded_output, padded_confidence


def test_upscale_flow():
    rngs = nnx.Rngs(0)
    flow_with_scores = jax.random.normal(rngs.flow(), (2, 8, 7, 5))
    upscaler = FlowUpscaler(rngs=rngs)
    upscaled_flow, confidence = upscaler(flow_with_scores)
    assert upscaled_flow.shape == (2, 16, 14, 2)
    assert confidence.shape == (2, 16, 14, 1)
