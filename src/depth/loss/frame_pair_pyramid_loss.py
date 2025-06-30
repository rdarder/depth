from typing import Sequence, Any

import jax
import jax.numpy as jnp
from flax import nnx

from depth.loss.flow_pyramid_loss import calc_flow_pyramid_loss
from depth.model.pyramid_flow import PyramidFlowEstimator


def frame_pair_pyramid_loss(model: PyramidFlowEstimator,
                            pyramid1: Sequence[jax.Array],
                            pyramid2: Sequence[jax.Array],
                            priors: jax.Array) -> tuple[jax.Array, Sequence[dict[str, Any]]]:
    flow_pyramid, aux_pyramid = model(pyramid1, pyramid2, priors)
    flow_pyramid_loss = calc_flow_pyramid_loss(flow_pyramid, aux_pyramid)
    level_losses = jnp.array([jnp.mean(level) for level in flow_pyramid_loss])
    weights = jnp.array([1., 0.5] + [0] * (len(level_losses) - 2))
    weights = weights / jnp.sum(weights)
    weighted_loss = jnp.sum(level_losses * weights)
    loss_aux = [
        dict(**flow_aux, loss=loss, loss_grid=loss_grid, loss_weight=weight, flow=flow) for
        flow, flow_aux, loss, loss_grid, weight in
        zip(flow_pyramid, aux_pyramid, level_losses, flow_pyramid_loss, weights)
    ]
    return weighted_loss, loss_aux


frame_pair_pyramid_loss_value_and_grad = nnx.value_and_grad(frame_pair_pyramid_loss, has_aux=True)
