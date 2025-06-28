import jax
import jax.numpy as jnp

from depth.patches.difference import sum_of_absolute_differences
from depth.patches.shift import shifted_center_patch


def patch_flow_loss(patch1: jax.Array, patch2: jax.Array, flow: jax.Array) -> jax.Array:
    """Evaluate how similar two patches are after shifting patch2 by the given flow amount.

    0 means minimum loss (max similarity), and 1 means max loss (min similarity). It assumes
    patch pixel values are between 0 and 1.

    Internally it splits in half, shifting patch2 halfway in the flow direction and
    patch1 halfway in the opposite direction. The difference is evaluated only on the center
    portion of the patches (for a 4x4 patch, the center shifted parts of size 3x3 will be
    compared).

    For multichannel patches, it returns the mean difference across channels.
    """

    assert flow.shape == (2,)
    assert patch1.shape == patch2.shape
    patch1_center = shifted_center_patch(patch1, -flow / 2)
    patch2_shifted = shifted_center_patch(patch2, flow / 2)
    return jnp.mean(sum_of_absolute_differences(patch1_center, patch2_shifted))


def test_patch_flow_loss_exact_match():
    canvas = jax.random.normal(jax.random.key(1), (5, 5, 1))
    patch1 = canvas[:4, :4, :]
    patch2 = canvas[1:, 1:, :]
    flow_loss = patch_flow_loss(patch1, patch2, jnp.array([-1., -1.]))
    assert jnp.allclose(flow_loss, 0, atol=1e-6)


def test_patch_flow_loss_close_match():
    canvas = jax.random.normal(jax.random.key(1), (5, 5, 1))
    patch1 = canvas[:4, :4, :]
    patch2 = canvas[1:, 1:, :]
    assert 0 < patch_flow_loss(patch1, patch2, jnp.array([-0.9, -0.9])) < 0.1


def test_patch_flow_loss_non_exact_match():
    canvas = jax.random.uniform(jax.random.key(1), (5, 5, 1))
    patch1 = canvas[:4, :4, :]
    patch2 = canvas[1:, 1:, :]
    assert 0.4 < patch_flow_loss(patch1, patch2, jnp.array([1.0, 1.0])) <= 1


def test_patch_flow_loss_exact_match_two_channels():
    canvas = jax.random.normal(jax.random.key(1), (5, 5, 2))
    patch1 = canvas[:4, :4, :]
    patch2 = canvas[1:, 1:, :]
    loss = patch_flow_loss(patch1, patch2, jnp.array([-1., -1.]))
    assert jnp.allclose(loss, 0, atol=1e-6)
