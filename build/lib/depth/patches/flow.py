import jax
import jax.numpy as jnp

from patches.loss_grid import aggregate_patch_grid_flow_loss_with_grad


def estimate_flow_grid(patches1: jax.Array, patches2: jax.Array, priors: jax.Array) -> jax.Array:
    """Estimate the flow between patches given a prior.

    The patches are arranged in a grid for convenience, but not necessary for the estimation.
    They can perfectly be all in the batch dimension if preferred. The grid dimensions
    are still required though.

    Patches dimensions:
        Batch, Channel, PatchesY, PatchesX, PatchWidth, PatchHeight
    """
    assert patches1.ndim == 6  # B, C, PY, PX, PH, PW
    assert patches1.shape == patches2.shape
    for i in range(10):
        loss0, grad0 = aggregate_patch_grid_flow_loss_with_grad(
            patches1,
            patches2,
            priors,
        )
        priors -= jnp.clip(grad0 / (10 * loss0 + 1e-6), -1, 0.1)
    return priors
