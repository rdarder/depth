from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import register_dataclass

from depth.loss.flow_pyramid_loss import calc_flow_pyramid_loss
from depth.model.pyramid_flow import PyramidFlowEstimator, PyramidFlowEstimationParams, \
    PyramidFlowEstimation


@register_dataclass
@dataclass
class LossTrace:
    losses: Sequence[jax.Array]
    flow: PyramidFlowEstimation
    avg_level_losses: jax.Array
    weights: jax.Array
    weighted_loss: jax.Array


def frame_pair_pyramid_loss(model: PyramidFlowEstimator,
                            pyramid1: Sequence[jax.Array],
                            pyramid2: Sequence[jax.Array],
                            priors: jax.Array) -> tuple[jax.Array, LossTrace]:
    pyramid_flow_params = PyramidFlowEstimationParams(
        pyramid1=pyramid1,
        pyramid2=pyramid2,
        prior=priors,
    )
    flow = model(pyramid_flow_params)
    level_losses = calc_flow_pyramid_loss(flow)
    avg_level_losses = jnp.array([jnp.mean(level) for level in level_losses])
    weights = 1 / jnp.array([2**i for i in range(len(avg_level_losses))])
    weights = weights / jnp.sum(weights)
    weighted_loss = jnp.sum(avg_level_losses * weights)
    loss_trace = LossTrace(
        losses=level_losses,
        flow=flow,
        avg_level_losses=avg_level_losses,
        weights=weights,
        weighted_loss=weighted_loss,
    )
    return weighted_loss, loss_trace


frame_pair_pyramid_loss_value_and_grad = nnx.value_and_grad(frame_pair_pyramid_loss, has_aux=True)
