# def get_patch_coordinates(
#     H: int, W: int, r_center: jax.Array, c_center: jax.Array, patch_size: int
# ):
#     """
#     Calculates top-left coordinates for dynamic_slice given center coordinates.
#     Also returns valid mask for patches that are fully within image.
#     (For initial simple version, we might not use mask and just clip/rely on dynamic_slice behavior)
#     """
#     patch_radius = patch_size // 2
#     # Calculate top-left for dynamic_slice
#     # We need to ensure start indices are non-negative.
#     # And that r_center + patch_radius < H, c_center + patch_radius < W
#     # For dynamic_slice, the start indices are r_center - patch_radius
#     start_r = r_center - patch_radius
#     start_c = c_center - patch_radius
#     # Simple clipping for start_indices for dynamic_slice
#     # More robust handling might be needed if patches can go way off image
#     # dynamic_slice expects start_indices to be within [0, dim_size - slice_size]
#     # However, if we pad the image, this becomes simpler.
#     # For now, let's assume we'll clip coordinates passed to dynamic_slice
#     # so that the *centers* are valid, and rely on dynamic_slice to handle edges if needed.
#     # Or, better, clip the *start_indices* so dynamic_slice is always valid.
#     # Start indices for dynamic_slice should be within [0, dim_size - patch_size]
#     # Let's adjust based on this for now.
#     # This is a bit tricky without padding.
#     # An easier way for initial step is to pad the image.
#     # Returning center coords for now, actual slicing will be in vmapped function
#     return r_center, c_center
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from incremental import hierarchical_flow_estimation
from predictor import MinimalPredictor
from pyramid import BaselinePyramid


def extract_patches_vectorized(
    image_batch: jax.Array,  # [B, H, W, C]
    batch_indices: jax.Array,  # [N_patches]
    r_centers: jax.Array,  # [N_patches]
    c_centers: jax.Array,  # [N_patches]
    patch_size: int,
) -> jax.Array:  # [N_patches, patch_size, patch_size, C]
    """
    Extracts patches using jax.lax.dynamic_slice, vectorized over patches.
    Assumes r_centers, c_centers are already clipped to valid *center* ranges.
    """
    B, H, W, C = image_batch.shape
    patch_radius = patch_size // 2

    # Calculate top-left corner for dynamic_slice
    # These r_starts and c_starts are the crucial inputs for dynamic_slice
    r_starts = r_centers - patch_radius
    c_starts = c_centers - patch_radius

    # Clip start coordinates to ensure the slice itself is valid
    # dynamic_slice requires: 0 <= start_index <= dim_size - slice_size
    r_starts_clipped = jnp.clip(r_starts, 0, H - patch_size)
    c_starts_clipped = jnp.clip(c_starts, 0, W - patch_size)

    # We need to vmap the slicing operation.
    # jax.lax.dynamic_slice(operand, start_indices, slice_sizes)
    # operand is image_batch[b_idx]
    # start_indices is (r_start_clipped, c_start_clipped, 0) for channel
    # slice_sizes is (patch_size, patch_size, C)

    def _slice_one_patch(image_single_batch, r_start, c_start):
        # image_single_batch: [H, W, C]
        # r_start, c_start: scalar
        return jax.lax.dynamic_slice(
            image_single_batch,
            (r_start, c_start, 0),  # start_indices for H, W, C
            (patch_size, patch_size, C),  # slice_sizes for H, W, C
        )

    # To vmap this, we need to vmap over (image_batch[b_idx], r_starts_clipped, c_starts_clipped)
    # This is slightly tricky because of the batch_indices.
    # An alternative is to vmap over a per-patch function that indexes into the batch.

    # Let's make a vmappable function that takes b, r_start, c_start
    def _slice_for_vmap(b_idx, r_start_clipped, c_start_clipped):
        return _slice_one_patch(image_batch[b_idx], r_start_clipped, c_start_clipped)

    all_patches = jax.vmap(_slice_for_vmap)(
        batch_indices, r_starts_clipped, c_starts_clipped
    )
    return all_patches


def compute_photometric_loss(
    frame1_original: jax.Array,  # [B, H_0, W_0, 1]
    frame2_original: jax.Array,  # [B, H_0, W_0, 1]
    predicted_flow_level0: jax.Array,  # [B, H_0/2, W_0/2, 2]
    patch_size: int,
    loss_type: str = "l1",
) -> jax.Array:
    """
    Computes photometric loss between frame1 and warped frame2 at Level 0.
    Uses rounded coordinates for warping (no interpolation for P2).
    """
    B, H0, W0, C = frame1_original.shape
    _, HP, WP, _ = predicted_flow_level0.shape
    assert C == 1  # Expect grayscale
    assert HP == H0 // 2
    assert WP == W0 // 2

    # Create a grid of all (b, r, c) locations for dense loss
    batch_coords, r_coords_grid, c_coords_grid = jnp.meshgrid(
        jnp.arange(B),
        jnp.arange(H0, step=2),  # All r coordinates
        jnp.arange(W0, step=2),  # All c coordinates
        indexing="ij",
    )

    # Flatten to get lists of coordinates for vmapping
    b_indices_flat = batch_coords.flatten()  # [B*H0*W0]
    r_indices_flat = r_coords_grid.flatten().astype(jnp.int32)  # [B*H0*W0]
    c_indices_flat = c_coords_grid.flatten().astype(jnp.int32)  # [B*H0*W0]

    patches_p1 = extract_patches_vectorized(
        frame1_original, b_indices_flat, r_indices_flat, c_indices_flat, patch_size
    )

    # 2. Calculate warped coordinates for P2
    flow_at_locs = predicted_flow_level0[
        b_indices_flat, r_centers_flat, c_centers_flat, :
    ]  # [B*H0*W0, 2]

    target_r_f2_float = (
        r_centers_flat.astype(jnp.float32) + flow_at_locs[:, 1]
    )  # flow_uy
    target_c_f2_float = (
        c_centers_flat.astype(jnp.float32) + flow_at_locs[:, 0]
    )  # flow_ux

    rounded_target_r_f2 = jnp.round(target_r_f2_float).astype(jnp.int32)
    rounded_target_c_f2 = jnp.round(target_c_f2_float).astype(jnp.int32)

    # Clip these rounded *center* coordinates before passing to extract_patches_vectorized
    # to ensure centers are within [0, Dim-1]
    clipped_rounded_target_r_f2 = jnp.clip(rounded_target_r_f2, 0, H0 - 1)
    clipped_rounded_target_c_f2 = jnp.clip(rounded_target_c_f2, 0, W0 - 1)

    # Extract patches P2 from frame2_original using these warped & rounded coordinates
    patches_p2 = extract_patches_vectorized(
        frame2_original,
        b_indices_flat,
        clipped_rounded_target_r_f2,
        clipped_rounded_target_c_f2,
        patch_size,
    )

    # 3. Calculate loss
    patch_diff = patches_p1 - patches_p2  # [N_total_patches, patch_size, patch_size, 1]

    if loss_type == "l1":
        loss_values = jnp.abs(patch_diff)
    elif loss_type == "l2":
        loss_values = jnp.square(patch_diff)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Average loss over patch pixels, then over all patches in the batch
    loss_per_patch = jnp.mean(
        loss_values, axis=(1, 2, 3)
    )  # Avg over H_patch, W_patch, C_patch
    total_loss = jnp.mean(loss_per_patch)  # Avg over all patches

    return total_loss


# Example Usage (requires dummy hierarchical_flow_estimation and its support modules)
if __name__ == "__main__":
    # Re-use dummy setup from previous hierarchical_flow_estimation example
    key = jax.random.PRNGKey(0)
    rngs_pyramid = nnx.Rngs(params=key)
    rngs_predictor = nnx.Rngs(params=jax.random.PRNGKey(1))

    num_pyr_levels = 3
    img_h0, img_w0 = 32, 32  # Must be divisible by 2^(num_pyr_levels-1)
    batch_size = 2

    dummy_pyramid_builder = BaselinePyramid(
        num_levels=num_pyr_levels, rngs=rngs_pyramid
    )
    dummy_predictor = MinimalPredictor(rngs=rngs_predictor)

    frame1 = jnp.arange(batch_size * img_h0 * img_w0 * 1, dtype=jnp.float32).reshape(
        batch_size, img_h0, img_w0, 1
    ) / (batch_size * img_h0 * img_w0)
    # Frame2 is Frame1 shifted by (1,1) pixel ideally, and with some value changes
    frame2 = jnp.roll(frame1, shift=(0, 1, 1, 0), axis=(0, 1, 2, 3)) + 0.1
    frame2 = jnp.clip(frame2, 0, 1.0)

    pyramid_f1 = dummy_pyramid_builder(frame1)
    pyramid_f2 = dummy_pyramid_builder(frame2)

    predicted_flow = hierarchical_flow_estimation(
        pyramid_f1, pyramid_f2, dummy_predictor, num_pyr_levels
    )

    print(f"Frame1 shape: {frame1.shape}")
    print(f"Frame2 shape: {frame2.shape}")
    print(f"Predicted flow shape: {predicted_flow.shape}")

    patch_s = 5
    loss_l1 = compute_photometric_loss(
        frame1, frame2, predicted_flow, patch_size=patch_s, loss_type="l1"
    )
    loss_l2 = compute_photometric_loss(
        frame1, frame2, predicted_flow, patch_size=patch_s, loss_type="l2"
    )

    print(f"L1 Loss (patch {patch_s}x{patch_s}): {loss_l1}")
    print(f"L2 Loss (patch {patch_s}x{patch_s}): {loss_l2}")

    # Test with a known flow: if flow is (1,1) and frame2 is frame1 shifted by (1,1)
    # then loss should be small (due to +0.1 in frame2)
    # If predictor outputs zero flow, it means P2 is taken from unshifted frame2.
    # P1 from frame1 at (r,c), P2 from frame2 at (r,c)
    # Since frame2 is frame1 shifted by (1,1) + 0.1, P2 from (r,c) is like P1 from (r-1, c-1) + 0.1
    # This will result in some non-zero loss.

    # If predicted_flow was exactly the true shift (e.g., all ones if shift is (1,1))
    # then P2 would be sampled from frame2 at (r+1, c+1).
    # frame2[b, r+1, c+1] = frame1[b, r, c] + 0.1
    # So patch_diff would be approx -0.1. L1 loss per pixel ~0.1.

    known_flow_true_shift = jnp.ones_like(predicted_flow)  # Assuming (dx=1, dy=1)
    loss_l1_known_flow = compute_photometric_loss(
        frame1, frame2, known_flow_true_shift, patch_size=patch_s, loss_type="l1"
    )
    print(f"L1 Loss with 'true' (1,1) flow: {loss_l1_known_flow}")
    # Expected: approx 0.1 because frame2 = roll(frame1) + 0.1
