import jax
import jax.numpy as jnp

from depth.patches.loss import patch_flow_loss


def patch_match_score_loss(l_patch, l_mid, l=2.0):
    return 1 - 2 / (1 + jnp.exp(l * (jnp.abs(l_patch - l_mid)) / (l_mid + 1e-6)))


def patch_flow_loss_grid(patches1: jax.Array, patches2: jax.Array, flow: (jax.Array),
                         match_score: jax.Array) -> (
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
    mean_loss = jnp.mean(flat_losses, axis=-1)
    match_score_loss = patch_match_score_loss(flat_losses, mean_loss)
    compound_loss = 1.0 * flat_losses + 0.1 * match_score_loss
    return compound_loss.reshape(B, PY, PX)
