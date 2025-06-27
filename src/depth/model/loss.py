from typing import Sequence, Any

import jax
import jax.numpy as jnp

from depth.model.multi_level_flow import PyramidFlowEstimator


def frame_pair_loss(model: PyramidFlowEstimator,
                    pyramid1: Sequence[jax.Array],
                    pyramid2: Sequence[jax.Array],
                    priors: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
    flow_pyramid_with_loss = model(pyramid1, pyramid2, priors)
    level_losses = jnp.array([jnp.mean(level[:, :, :, 2]) for level in flow_pyramid_with_loss])
    weighted_loss = level_losses[0]
    aux = dict(pyramid1=pyramid1,
               pyramid2=pyramid2,
               flow_with_loss=flow_pyramid_with_loss,
               levels_weights=jnp.array([1.0]),
               levels_losses=level_losses)

    return weighted_loss, aux
