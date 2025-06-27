import jax
import jax.numpy as jnp

from depth.patches.loss import patch_flow_loss


def patch_flow_loss_grid(patches1: jax.Array, patches2: jax.Array, flow: (jax.Array)) -> (
        jax.Array):
    """Calculates the patch loss over a grid of patches.

    Mostly a convenience function for processing patches of an image while keeping the patch
    spatial relationship in the parameters and return shapes.
    """
    # patches1, patches2: [B, PY, PX, PH, PW, C]
    # flow: [B, PY, PX, 2]
    # return [B, PY, PX]
    assert patches1.shape == patches2.shape
    B, PY, PX, PH, PW, C = patches1.shape
    FB, FY, FX, FC = flow.shape
    assert FB == B
    assert FC == 2
    assert FY == PY
    assert FX == PX
    flat_patches1 = patches1.reshape(-1, PH, PW, C)
    flat_patches2 = patches2.reshape(-1, PH, PW, C)
    flat_flow = flow.reshape(-1, 2)
    flat_losses = jax.vmap(patch_flow_loss)(flat_patches1, flat_patches2, flat_flow)
    return flat_losses.reshape(B, PY, PX)
