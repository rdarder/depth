import jax
import jax.numpy as jnp

from depth.model.frame_pair_flow import FramePairFlow


def frame_pair_loss(model: FramePairFlow, f1: jax.Array, f2: jax.Array, priors: jax.Array):
    flow_pyramid_with_loss, img_pyramid1, img_pyramid2 = model(f1, f2, priors)
    loss = jnp.mean(flow_pyramid_with_loss[0][:,:,:,2])
    aux = dict(pyramid1=img_pyramid1, pyramid2=img_pyramid2, flow_with_loss=flow_pyramid_with_loss)
    return loss, aux
