from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.ndimage

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


def get_batched_patches_differentiable(
    source: jax.Array, coords_float_top_left: jax.Array, patch_size: int
) -> jax.Array:
    """
    Extracts patches from a batch of source arrays at specified BATCED FLOATING POINT coordinates
    using differentiable sampling (map_coordinates).

    Assumes source has shape (B, H, W, ...) and coords_float_top_left has shape (B, N, 2) (float).
    The coordinates in 'coords_float_top_left' represent the TOP-LEFT corner of the desired patch.

    Args:
        source: The source JAX array batch. Shape (B, H, W, ...).
        coords_float_top_left: A batch of (row, column) FLOATING POINT coordinates for the TOP-LEFT of the patch. Shape (B, N, 2).
        patch_size: The size of the square patch to extract.

    Returns:
        A JAX array containing the extracted patches for all batch elements
        and all coordinate locations. Shape (B, N, patch_size, patch_size, ...).
        The trailing dimensions (...) will match those of the source array.
    """

    def _get_single_patch_from_image_differentiable(
        image, patch_top_left_coord_float, size
    ):
        # image: (H, W, C) or (H, W)
        # patch_top_left_coord_float: (2,) - (r, c) float for the top-left corner
        # size: P (patch_size)

        # Generate sampling coordinates for this patch grid relative to the top-left corner
        # The sampling points are pixel centers, so offsets are 0.0, 1.0, ..., size-1.0
        dr = jnp.arange(size, dtype=jnp.float32)  # 0.0, 1.0, ..., patch_size-1.0
        dc = jnp.arange(size, dtype=jnp.float32)  # 0.0, 1.0, ..., patch_size-1.0
        dr_grid, dc_grid = jnp.meshgrid(dr, dc, indexing="ij")  # (P, P)
        # Stack into (dr, dc) pairs: (P, P, 2)
        offsets = jnp.stack([dr_grid, dc_grid], axis=-1)  # (P, P, 2)

        # Add patch top-left coordinate to offsets to get absolute sampling coordinates in image grid
        # Broadcasts (P, P, 2) + (2,) -> (P, P, 2)
        # These are the (row, col) floating point coordinates for each pixel within the patch relative to the image origin.
        sampling_coords = patch_top_left_coord_float + offsets  # (P, P, 2) floats

        # map_coordinates expects coordinates in a specific format.
        # For a sequence of arrays, it expects a list/tuple where each element
        # is a 1D array of coordinates for that dimension.
        # Our `sampling_coords` is (P, P, 2). We need to reshape to (P*P, 2) and then split.
        sampling_coords_flat = sampling_coords.reshape(-1, 2)  # Shape (P*P, 2)

        # Split the (P*P, 2) array into two (P*P,) arrays (one for rows, one for columns)
        # jnp.split splits along the specified axis. splitting (P*P, 2) along axis 1 gives list of 2 arrays (P*P, 1).
        # We need to split along axis 1 and squeeze the resulting arrays.
        coords_r_flat, coords_c_flat = jnp.split(
            sampling_coords_flat, 2, axis=1
        )  # Each shape (P*P, 1)
        # Squeeze the singleton dimension
        coords_r_flat = jnp.squeeze(coords_r_flat, axis=1)  # Shape (P*P,)
        coords_c_flat = jnp.squeeze(coords_c_flat, axis=1)  # Shape (P*P,)

        # Create the sequence of coordinate arrays: (r_coords, c_coords)
        sampling_coords_sequence = (
            coords_r_flat,
            coords_c_flat,
        )  # Tuple of 2 arrays, each shape (P*P,)

        # jax.scipy.ndimage.map_coordinates samples at the specified coordinates.
        # The output shape will be (num_points,) + input.shape[input_rank:]
        # num_points = P*P, input_rank = 2 (for H, W)
        # If image is (H, W), output is (P*P,)
        # If image is (H, W, C), output is (P*P, C)
        sampled_values_flat = jax.scipy.ndimage.map_coordinates(
            jnp.squeeze(image),
            sampling_coords_sequence,  # Pass the sequence of arrays
            order=1,  # Use linear interpolation
            mode="nearest",  # How to handle points outside bounds (nearest neighbor value)
            # Other modes like 'wrap', 'reflect', 'constant' are also options.
        )  # Shape (P*P,) or (P*P, C)

        # Reshape sampled_values back to (P, P, C) or (P, P) spatial shape
        # The output of map_coordinates does NOT put the feature dimension first when coordinates is a sequence.
        if image.ndim == 3:  # Source has (H, W, C)
            # (P*P, C) -> reshape to (P, P, C)
            sampled_values_reshaped = sampled_values_flat.reshape(
                size, size, image.shape[-1]
            )
        else:  # Source has (H, W) - grayscale (ndim == 2)
            # (P*P,) -> reshape to (P, P)
            sampled_values_reshaped = sampled_values_flat.reshape(size, size)

        return sampled_values_reshaped  # Shape (P, P, C) or (P, P)

    # Vmap the inner function over the Batch dimension (B) and the Coordinate dimension (N).
    # We need a nested vmap structure:
    # 1. Vmap _get_single_patch_from_image_differentiable over the N dimension of coords_single_float for one image.
    # 2. Vmap the result over the B dimension of source and coords_float_top_left.

    def _get_all_patches_from_image_differentiable(
        image_single, coords_single_float, size
    ):
        # image_single: (H, W, C) or (H, W)
        # coords_single_float: (N, 2) float - these are the N top-left patch coordinates for this image
        # size: P
        # Vmap _get_single_patch_from_image_differentiable over the N dimension of coords_single_float
        get_patches_vmapped_over_coords = jax.vmap(
            _get_single_patch_from_image_differentiable, in_axes=(None, 0, None)
        )
        return get_patches_vmapped_over_coords(
            image_single, coords_single_float, size
        )  # Output shape: (N, P, P, C) or (N, P, P)

    # Vmap _get_all_patches_from_image_differentiable over the Batch dimension
    # source: (B, H, W, C) -> axis 0
    # coords_float_top_left: (B, N, 2) -> axis 0
    # size: None (static)
    get_batched_patches_vmapped = jax.vmap(
        _get_all_patches_from_image_differentiable, in_axes=(0, 0, None)
    )

    # Call the vmapped function with the batched inputs.
    patches = get_batched_patches_vmapped(source, coords_float_top_left, patch_size)

    return patches  # Final shape: (B, N, P, P, C) or (B, N, P, P)


def photometric_loss_for_level(
    frame1: jax.Array,  # [B, H, W, 1]
    frame2: jax.Array,  # [B, H, W, 1]
    f1_coords,  # [B, N, 2] dtype=int
    f2_coords,  # [B, N, 2] dtype=float
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
    f2_patches = get_batched_patches_differentiable(frame2, f2_coords, patch_size)
    patch_losses = jnp.abs(f1_patches - f2_patches)
    per_batch_loss_sum = jnp.sum(patch_losses, axis=(1, 2, 3, 4))
    return per_batch_loss_sum
    # masked_loss = jnp.where(overflow_mask, per_patch_loss, 0)
    # in_frame = jnp.count_nonzero(overflow_mask, axis=1)
    # loss = jnp.sum(masked_loss, axis=1)
    # return in_frame, loss


def level_loss_discount(previous_level_loss, level_loss):
    previous = jax.lax.stop_gradient(previous_level_loss)
    current = jax.lax.stop_gradient(level_loss)
    loss_ratio = jnp.clip(previous / (1.0 * current + 1e-8), max=jnp.array(1.0))
    return loss_ratio

def photometric_loss(f1_pyramid, f2_pyramid, estimations, patch_size):
    coarsest_frame = f1_pyramid[0]
    B, H, W, C = coarsest_frame.shape
    level_loss_weight_b = jnp.ones(B, dtype=jnp.float32)
    previous_level_loss_avg_b = jnp.ones(B, dtype=jnp.float32)
    loss_b = jnp.array(0.0)
    debug_info_levels = []
    total_pixel_count = 0
    for level in range(len(f1_pyramid)):
        f1_coords = estimations["f1_kept"][level] * 2
        f2_coords = estimations["f2_kept"][level] * 2
        level_loss_sum_b = photometric_loss_for_level(
            frame1=f1_pyramid[level],
            frame2=f2_pyramid[level],
            f1_coords=f1_coords,
            f2_coords=f2_coords,
            flow_with_confidence=estimations["flow_with_confidence"][level],
            patch_size=patch_size,
        )
        pixel_count = (f1_coords.shape[1] * f1_coords.shape[0])
        total_pixel_count += pixel_count
        # batched_level_loss = level_loss / jnp.clip(level_in_frame, 1)
        level_pixel_loss_avg = level_loss_sum_b / pixel_count
        weighted_level_loss_sum = level_loss_sum_b * level_loss_weight_b
        next_level_loss_discount_b = level_loss_discount(previous_level_loss_avg_b, level_pixel_loss_avg)
        level_loss_weight_b = level_loss_weight_b * next_level_loss_discount_b
        previous_level_loss_avg_b =  level_pixel_loss_avg

        weighted_level_loss_avg = jnp.mean(weighted_level_loss_sum)
        level_loss_weight_avg = jnp.mean(level_loss_weight_b)
        # jax.debug.print("weighted_level_loss_avg: {x}", x=weighted_level_loss_avg)
        # jax.debug.print("next_level_loss_discount_avg: {x}", x=jnp.mean(next_level_loss_discount_b))
        # jax.debug.print("level_loss_weight_avg: {x}", x=level_loss_weight_avg)

        loss_b += weighted_level_loss_sum
        debug_info_levels.append({
            "level_idx": level,
            "weighted_loss_avg": weighted_level_loss_avg,
            "loss_weight_avg": level_loss_weight_avg,
            "loss": weighted_level_loss_avg,
            })

    # dividing by the last loss discount b effectively raises up the loss count for the
    # last level the equivalent of not being weighted, and lifts all of the other levels
    # proportionally. It's awful, The goal is not to incentivize worsening at the top levels
    # because that would reduce the overall loss. Instead the upper levels are penalized more
    # based on the last levels performance.
    loss = jnp.mean((loss_b / level_loss_weight_b) / total_pixel_count)

    debug_info = {
        "per_level": debug_info_levels,
        "loss": loss,
    }
    return loss, debug_info


def model_loss(
    model: OpticalFlow,
    batch_frame1: jax.Array,
    batch_frame2: jax.Array,
    patch_size: int,
    priors: Optional[jax.Array],
    flow_regularization_weight: float = 1e-5,
):
    """Computes the loss for gradient calculation."""
    # Forward pass through the main model
    f1_pyramid, f2_pyramid, estimations = model(
        batch_frame1, batch_frame2, priors
    )
    levels = len(f1_pyramid)
    f1_source = [conv_output[:, :, :, :1] for conv_output in f1_pyramid[1:levels]] + [
        batch_frame1
    ]
    f2_source = [conv_output[:, :, :, :1] for conv_output in f2_pyramid[1:levels]] + [
        batch_frame2
    ]

    last_level_flow = estimations["flow_with_confidence"][-1][:, :, :2]
    avg_flow = jnp.average(last_level_flow**2)

    # Compute photometric loss
    photo_loss, photo_debug_info = photometric_loss(
        f1_source,
        f2_source,
        estimations,
        patch_size=patch_size,
    )
    flow_regularization_loss = avg_flow * flow_regularization_weight
    total_loss = photo_loss + flow_regularization_loss

    debug_info = {
        "total_loss": total_loss,
        "photometric_loss_component": photo_loss,
        "flow_regularization_loss_component": flow_regularization_loss,
        "avg_flow_squared": avg_flow,
        "photometric": photo_debug_info # Nested debug info
    }

    return total_loss, debug_info