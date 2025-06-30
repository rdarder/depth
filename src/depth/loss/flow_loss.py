#         loss = patch_flow_loss_grid(patches1, patches2, remainder_flow)[:, :, :, None]
import jax
import jax.numpy as jnp
from flax import nnx

from depth.model.patch_flow import PatchFlowEstimator
from depth.model.single_level_flow import LevelFlowEstimator
from depth.patches.loss_grid import patch_flow_loss_grid


def calc_flow_loss(flow_with_scores: jax.Array, aux: dict) -> jax.Array:
    """Return the flow loss between patches of frame1 and frame2.
    shape of f1, f2: (B, H, W, C)
    shape of priors: (B, PY, PX, 2)
    shape of flow: (B, PY, PX, 2)
    where PY = conv_output_size(H, patch_size, patch_stride)
    """
    flow, confidence, patch_score = jnp.split(flow_with_scores, (2, 3), axis=-1)
    remainder_flow = flow - jnp.round(flow)
    loss = patch_flow_loss_grid(aux['patches1'], aux['patches2'], remainder_flow)
    return loss


def test_single_level_flow():
    rngs = nnx.Rngs(0)
    img = jax.random.uniform(jax.random.key(1), (3, 6, 8, 2))
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=2, train=False, rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)
    prior = jax.random.uniform(jax.random.key(2), (3, 2, 3, 2))
    flow_with_scores, aux = level_flow_estimator(img, img, prior)
    loss = calc_flow_loss(flow_with_scores, aux)
    assert loss.shape == (3, 2, 3)
