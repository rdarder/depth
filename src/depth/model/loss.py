import jax
import jax.numpy as jnp
from jax.nn import softmax

from depth.model.frame_pair_flow import FramePairFlow


def frame_pair_loss(model: FramePairFlow, f1: jax.Array, f2: jax.Array, priors: jax.Array):
    flow_pyramid_with_loss, img_pyramid1, img_pyramid2 = model(f1, f2, priors)
    level_losses = jnp.array([jnp.mean(level[:, :, :, 2]) for level in flow_pyramid_with_loss])
    ratio_numerators = level_losses[:-1]
    ratio_denominators = level_losses[1:] + 1e-6
    level_losses_ratios = ratio_numerators / ratio_denominators
    level_losses_ratios_stopped = jax.lax.stop_gradient(level_losses_ratios)
    level_losses_ratios_extended = jnp.concatenate(
        [level_losses_ratios_stopped, level_losses_ratios_stopped[-1:] / 2]
    )
    weights = level_losses_ratios_extended / jnp.sum(level_losses_ratios_extended)
    weighted_loss = jnp.sum(level_losses * weights)
    aux = dict(pyramid1=img_pyramid1,
               pyramid2=img_pyramid2,
               flow_with_loss=flow_pyramid_with_loss,
               levels_weights=weights,
               levels_losses=level_losses)

    return weighted_loss, aux
