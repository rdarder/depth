from typing import Optional

import jax
import jax.numpy as jnp

from flow_model import OpticalFlow


def get_patch_overflow_mask(
    patch_coords: jax.Array, height: int, width: int, patch_size: int
):
    max_valid_row = height - patch_size
    max_valid_col = width - patch_size
    r = patch_coords[:, :, 0]
    c = patch_coords[:, :, 1]
    return (r <= max_valid_row) & (c <= max_valid_col)


def get_batched_patches(
    source: jax.Array, coords: jax.Array, patch_size: int
) -> jax.Array:
    """
    Extracts patches from a batch of source arrays at specified batched coordinates.

    Assumes source has shape (B, H, W, ...) and coords has shape (B, N, 2).
    Does NOT check for out-of-bounds coordinates.

    Args:
        source: The source JAX array batch. Shape (B, H, W, ...).
        coords: A batch of (row, column) coordinates. Shape (B, N, 2).
        patch_size: The size of the square patch to extract.

    Returns:
        A JAX array containing the extracted patches for all batch elements
        and all coordinates. Shape (B, N, patch_size, patch_size, ...).
        The trailing dimensions (...) will match those of the source array (e.g., C).
    """

    def _get_single_patch(source_arr, single_coord, size):
        start_indices_rc = single_coord.astype(jnp.int32)
        num_spatial_dims = 2
        num_feature_dims = source_arr.ndim - num_spatial_dims
        start_indices_features = jnp.zeros(num_feature_dims, dtype=jnp.int32)
        start_indices = jnp.concatenate([start_indices_rc, start_indices_features])
        slice_sizes_spatial = (size, size)
        slice_sizes_features = source_arr.shape[num_spatial_dims:]
        slice_sizes = slice_sizes_spatial + slice_sizes_features
        slice_sizes_tuple = tuple(slice_sizes)
        return jax.lax.dynamic_slice(source_arr, start_indices, slice_sizes_tuple)

    def _process_single_batch_element(source_single, coords_single, size):
        get_patches_for_single_image = jax.vmap(
            _get_single_patch, in_axes=(None, 0, None)
        )
        return get_patches_for_single_image(
            source_single, coords_single, size
        )  # Output shape: (N, patch_size, patch_size, ...)

    get_batched_patches_vmapped = jax.vmap(
        _process_single_batch_element, in_axes=(0, 0, None)
    )
    patches = get_batched_patches_vmapped(source, coords, patch_size)
    return patches


def photometric_loss_for_level(
    frame1: jax.Array,  # [B, 2*H, 2*W, 1]
    frame2: jax.Array,  # [B, 2*H, 2*W, 1]
    f1_coords,
    f2_coords,
    flow_with_confidence: jax.Array,  # [B, H, W, 3]
    patch_size: int,
) -> jax.Array:
    """
    Computes photometric loss between frame1 and warped frame2 at Level 0.
    Uses rounded coordinates for warping (no interpolation for P2).
    """
    B, HF, WF, C = frame1.shape
    assert C == 1  # Expect grayscale
    assert f1_coords.shape == f2_coords.shape
    assert f1_coords.shape[:2] == flow_with_confidence.shape[:2]
    f1_overflow_mask = get_patch_overflow_mask(f1_coords, HF, WF, patch_size)
    f2_overflow_mask = get_patch_overflow_mask(f2_coords, HF, WF, patch_size)
    overflow_mask = f1_overflow_mask & f2_overflow_mask
    f1_patches = get_batched_patches(frame1, f1_coords, patch_size)
    f2_patches = get_batched_patches(frame2, f1_coords, patch_size)
    patch_losses = jnp.abs(f1_patches - f2_patches)
    per_patch_loss = jnp.mean(patch_losses, axis=(2, 3, 4))
    masked_loss = jnp.where(overflow_mask, per_patch_loss, 0)
    in_frame = jnp.count_nonzero(overflow_mask)
    loss = jnp.sum(masked_loss)
    return in_frame, loss


def photometric_loss(f1_pyramid, f2_pyramid, estimations, patch_size):
    coarsest_frame = f1_pyramid[0]
    B, H, W, C = coarsest_frame.shape
    sum_loss = jnp.array([0.0])
    sum_in_frame = jnp.array([0])
    for level in range(len(f1_pyramid)):
        f1_coords = estimations["f1_kept"][level] * 2
        f2_coords = estimations["f2_kept"][level] * 2
        level_in_frame, level_loss = photometric_loss_for_level(
            frame1=f1_pyramid[level],
            frame2=f2_pyramid[level],
            f1_coords=f1_coords,
            f2_coords=f2_coords,
            flow_with_confidence=estimations["flow_with_confidence"][level],
            patch_size=patch_size,
        )
        sum_loss = sum_loss + level_loss
        sum_in_frame = sum_in_frame + level_in_frame

    loss = sum_loss / jnp.clip(sum_in_frame, 1)
    return loss[0]


def model_loss(
    model: OpticalFlow,
    batch_frame1: jax.Array,
    batch_frame2: jax.Array,
    patch_size: int,
    priors: Optional[jax.Array] = None,
):
    """Computes the loss for gradient calculation."""
    # Forward pass through the main model
    f1_pyramid, f2_pyramid, predicted_flow = model(
        batch_frame1, batch_frame2, priors=priors
    )
    levels = len(f1_pyramid)
    f1_source = [conv_output[:, :, :, :1] for conv_output in f1_pyramid[1:levels]] + [
        batch_frame1
    ]
    f2_source = [conv_output[:, :, :, :1] for conv_output in f2_pyramid[1:levels]] + [
        batch_frame2
    ]

    # Compute photometric loss
    loss = photometric_loss(
        f1_source,
        f2_source,
        predicted_flow,
        patch_size=patch_size,
    )

    return loss
