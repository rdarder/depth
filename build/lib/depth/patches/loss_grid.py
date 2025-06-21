import jax
import jax.numpy as jnp

from patches.loss import patch_flow_loss


def patch_flow_loss_grid(patches1: jax.Array, patches2: jax.Array, flow: (jax.Array)) -> (
        jax.Array):
    """Calculates the patch loss over a grid of patches.

    Mostly a convenience function for processing patches of an image while keeping the patch
    spatial relationship in the parameters and return shapes.
    """
    # patches1, patches2: [B, C, PY, PX, PH, PW]
    # flow: [B, 2, PY, PX]
    # return [B, PY, PX]
    assert patches1.shape == patches2.shape
    B, C, PY, PX, PH, PW = patches1.shape
    FB, FC, FY, FX = flow.shape
    assert FB == B
    assert FC == 2
    assert FY == PY
    assert FX == PX
    flat_patches1 = (
        patches1.transpose(0, 2, 3, 1, 4, 5).reshape(B * PY * PX, C, PH, PW)
    )
    flat_patches2 = (
        patches2.transpose(0, 2, 3, 1, 4, 5).reshape(B * PY * PX, C, PH, PW)
    )
    flat_flow = flow.transpose(0, 2, 3, 1).reshape(B * PY * PX, FC)
    flat_losses = jax.vmap(patch_flow_loss)(flat_patches1, flat_patches2, flat_flow)
    return flat_losses.reshape(B, PY, PX)


def aggregate_patch_flow_loss_grid(patch_grid1: jax.Array, patch_grid2: jax.Array,
                                   flow_grid: jax.Array) -> jax.Array:
    """Calculates and aggregates the loss over a grid of patches.

    Mostly a convenience function for when the loss gradient is needed (gradients are
    only supported for scalar valued functions in jax).
    """
    loss = patch_flow_loss_grid(patch_grid1, patch_grid2, flow_grid)
    return jnp.sum(loss)


aggregate_patch_grid_flow_loss_with_grad = jax.value_and_grad(
    aggregate_patch_flow_loss_grid, argnums=2
)
