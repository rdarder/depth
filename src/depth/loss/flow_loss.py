import jax
import jax.numpy as jnp
from flax import nnx

from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimation, LevelFlowEstimationParams
from depth.model.single_level_flow import LevelFlowEstimator
from depth.patches.loss import patch_flow_loss


def patch_match_score_loss(l_patch, l_mid, l=2.0):
    return 1 / (1 + jnp.exp(l * (l_patch - l_mid) / (l_mid + 1e-6)))


def patch_flow_loss_grid(flow: LevelFlowEstimation) -> jax.Array:
    """Calculates the patch loss over a grid of patches.

    Mostly a convenience function for processing patches of an image while keeping the patch
    spatial relationship in the parameters and return shapes.
    """
    flow.check_shapes_consistent()
    B, PY, PX, PH, PW, C = flow.patches1.shape
    flat_patches1 = flow.patches1.reshape(-1, PH, PW, C)
    flat_patches2 = flow.patches2.reshape(-1, PH, PW, C)
    flat_flow = flow.net_flow.reshape(-1, 2)
    flat_losses = jax.vmap(patch_flow_loss)(flat_patches1, flat_patches2, flat_flow)
    mean_loss = jnp.mean(flat_losses, axis=-1)
    match_score_loss = patch_match_score_loss(flat_losses, mean_loss)
    compound_loss = 1.0 * flat_losses + 0.02 * match_score_loss
    return compound_loss.reshape(B, PY, PX)


def test_single_level_flow():
    rngs = nnx.Rngs(0)
    img = jax.random.uniform(jax.random.key(1), (3, 6, 8, 2))
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=2, train=False, rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)
    prior = jax.random.uniform(jax.random.key(2), (3, 2, 3, 2))
    params = LevelFlowEstimationParams(frame1=img, frame2=img, prior=prior)
    flow = level_flow_estimator(params)
    loss = patch_flow_loss_grid(flow)
    assert loss.shape == (3, 2, 3, 1)
