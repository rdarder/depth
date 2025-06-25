import jax
import jax.numpy as jnp


def normalized_cross_correlation(patch1, patch2, epsilon=1e-6) -> jax.Array:
    """Similarity metric between two patches with dimensions C,H,W"""
    H, W, C = patch1.shape
    assert H >= 3 and W >= 3  # need at least 3x3 for ncc
    assert patch1.shape == patch2.shape
    # implement normalized cross correlation
    mean1 = jnp.mean(patch1, axis=(0, 1))
    mean2 = jnp.mean(patch2, axis=(0, 1))
    patch1_centered = patch1 - mean1
    patch2_centered = patch2 - mean2
    numerator = jnp.sum(patch1_centered * patch2_centered, axis=(0, 1))
    denominator1 = jnp.sqrt(jnp.sum(patch1_centered ** 2, axis=(0, 1)) + epsilon)
    denominator2 = jnp.sqrt(jnp.sum(patch2_centered ** 2, axis=(0, 1)) + epsilon)
    denominator = (denominator1 * denominator2) + epsilon
    ncc_value = numerator / denominator
    inverted_ncc = 1 - ncc_value
    normalized_ncc = inverted_ncc / 2.0
    return jnp.clip(normalized_ncc, 0, 1)


def sum_of_absolute_differences(patch1: jax.Array, patch2: jax.Array) -> jax.Array:
    """Similarity metric between two patches with dimensions C,H,W"""
    H, W, C = patch1.shape
    pixel_sad = jnp.sum(jnp.abs(patch1 - patch2)) / (C * H * W)
    return pixel_sad


def test_ncc_same_patch_single_channel():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 3, 1))
    assert normalized_cross_correlation(sample_patch, sample_patch) < 1e-4


def test_ncc_same_patch_two_channels():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 3, 2))
    assert jnp.sum(normalized_cross_correlation(sample_patch, sample_patch)) < 1e-4


def test_altered_patch_ncc():
    patch1 = jax.random.normal(jax.random.key(1), (3, 3, 1))
    patch2 = patch1.at[0, 0, :].set(2.0)
    assert normalized_cross_correlation(patch1, patch2) > 0.1


def test_sad_same_patch_single_channel():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 3, 1))
    assert sum_of_absolute_differences(sample_patch, sample_patch) == 0


def test_sad_same_patch_dual_channel():
    sample_patch = jax.random.normal(jax.random.key(1), (3, 3, 2))
    jax.debug.print("{x}", x=sum_of_absolute_differences(sample_patch, sample_patch))
    assert jnp.all(sum_of_absolute_differences(sample_patch, sample_patch) == jnp.zeros((2,)))


def test_sad_altered_patch():
    patch1 = jax.random.normal(jax.random.key(1), (3, 3, 1))
    patch2 = patch1.at[0, :, 0].set(1.0)
    assert sum_of_absolute_differences(patch1, patch2) > 0.3
