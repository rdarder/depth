import jax
import jax.numpy as jnp

from predictor import MinimalPredictor
from pyramid import BaselinePyramid


# Define the helper function outside the main loss function
def get_single_patch_from_batch(
    image_all_batches, start_indices_for_one_patch, patch_size: int
):
    """
    Extracts a single patch from a batch of images using dynamic_slice.

    Args:
        image_all_batches: The full image batch tensor (batch, H_L0, W_L0, 1).
        start_indices_for_one_patch: The 4D start index for one patch [batch_idx, start_i, start_j, start_c].
        patch_size: The size of the square patch.

    Returns:
        A tensor of shape (1, patch_size, patch_size, 1) representing the extracted patch.
    """
    # slice_sizes must have length equal to the operand rank (4).
    # We want to slice 1 element along the batch dim, patch_size along H and W, and 1 along C.
    slice_sizes = (1, patch_size, patch_size, 1)  # Size in (Batch, H, W, C)
    return jax.lax.dynamic_slice(
        image_all_batches, start_indices_for_one_patch, slice_sizes
    )


def compute_photometric_loss(
    predicted_flow_Lpred,  # Predicted flow from MinimalPredictor (batch, H_pred, W_pred, 2)
    image1_L0,  # Original Frame 1 (batch, H_L0, W_L0, 1)
    image2_L0,  # Original Frame 2 (batch, H_L0, W_L0, 1)
    L_pred,  # Level index where flow was predicted (e.g., 3)
    patch_size=3,  # Size of the square patch (e.g., 3)
    loss_type="l1",  # 'l1' for MAE, 'l2' for MSE
):
    """
    Computes the patch-wise photometric loss between Frame 1 and warped Frame 2.

    Args:
        predicted_flow_Lpred: Tensor of shape (batch, H_pred, W_pred, 2).
        image1_L0: Original Frame 1 grayscale tensor (batch, H_L0, W_L0, 1).
        image2_L0: Original Frame 2 grayscale tensor (batch, H_L0, W_L0, 1).
        L_pred: The pyramid level index corresponding to predicted_flow_Lpred.
        patch_size: The side length of the square patches to extract.
        loss_type: The type of photometric loss ('l1' or 'l2').

    Returns:
        A scalar tensor representing the average photometric loss.
    """
    batch_size, H_pred, W_pred, _ = predicted_flow_Lpred.shape
    _, H_L0, W_L0, _ = image1_L0.shape
    scale = 2**L_pred
    patch_half_size = patch_size // 2  # Integer division

    # 1. Upscale flow
    upscaled_flow = predicted_flow_Lpred * scale  # (batch, H_pred, W_pred, 2)

    # 2. Create base coordinates grid at L_pred resolution
    grid_i, grid_j = jnp.meshgrid(jnp.arange(H_pred), jnp.arange(W_pred), indexing="ij")

    # 3. Upscale grid coordinates to L_0 resolution and broadcast
    upscaled_grid_i = jnp.broadcast_to(grid_i * scale, (batch_size, H_pred, W_pred))
    upscaled_grid_j = jnp.broadcast_to(grid_j * scale, (batch_size, H_pred, W_pred))

    # 4. Calculate potential centers for patches in L_0
    frame1_center_i = upscaled_grid_i
    frame1_center_j = upscaled_grid_j
    frame2_center_i = upscaled_grid_i + upscaled_flow[..., 0]
    frame2_center_j = upscaled_grid_j + upscaled_flow[..., 1]

    # 5. Calculate top-left coordinates for patches (rounded)
    frame1_start_i = jnp.round(frame1_center_i - patch_half_size).astype(jnp.int32)
    frame1_start_j = jnp.round(frame1_center_j - patch_half_size).astype(jnp.int32)
    frame2_start_i = jnp.round(frame2_center_i - patch_half_size).astype(jnp.int32)
    frame2_start_j = jnp.round(frame2_center_j - patch_half_size).astype(jnp.int32)

    # 6. Clamp top-left coordinates to stay within image bounds for slicing
    frame1_start_i = jnp.clip(frame1_start_i, 0, H_L0 - patch_size)
    frame1_start_j = jnp.clip(frame1_start_j, 0, W_L0 - patch_size)
    frame2_start_i = jnp.clip(frame2_start_i, 0, H_L0 - patch_size)
    frame2_start_j = jnp.clip(frame2_start_j, 0, W_L0 - patch_size)

    # 7. Prepare start indices for dynamic_slice using vmap
    # Stack indices: (batch, H_pred, W_pred, 4) where last dim is [batch_idx, start_i, start_j, start_channel]
    batch_indices = jnp.broadcast_to(
        jnp.arange(batch_size)[:, None, None], (batch_size, H_pred, W_pred)
    )
    channel_start_index = jnp.zeros_like(
        batch_indices
    )  # Starting channel index is always 0

    frame1_start_indices_vmap = jnp.stack(
        [batch_indices, frame1_start_i, frame1_start_j, channel_start_index], axis=-1
    )
    frame2_start_indices_vmap = jnp.stack(
        [batch_indices, frame2_start_i, frame2_start_j, channel_start_index], axis=-1
    )

    # Flatten the leading dimensions of the start indices for vmap
    frame1_start_indices_flat = frame1_start_indices_vmap.reshape(
        -1, 4
    )  # (total_patches, 4)
    frame2_start_indices_flat = frame2_start_indices_vmap.reshape(
        -1, 4
    )  # (total_patches, 4)

    # 8. Apply vmap to get all patches
    # vmap over the first dimension of the flattened start indices (in_axes=0).
    # image_L0 is not mapped over (in_axes=None).
    # patch_size is a static argument to get_single_patch_from_batch, passed as a static_argnums tuple.
    # static_argnums tells vmap that this argument is a Python value, not a JAX array to be mapped over.
    batched_get_patch = jax.vmap(get_single_patch_from_batch, in_axes=(None, 0, None))

    frame1_patches_flat = batched_get_patch(
        image1_L0, frame1_start_indices_flat, patch_size
    )
    frame2_patches_flat = batched_get_patch(
        image2_L0, frame2_start_indices_flat, patch_size
    )

    # Expected shape of frame*_patches_flat: (total_patches, 1, patch_size, patch_size, 1)

    # Reshape back to the desired output shape: (batch_size, H_pred, W_pred, patch_size, patch_size, 1)
    frame1_patches_reshaped = frame1_patches_flat.reshape(
        batch_size, H_pred, W_pred, 1, patch_size, patch_size, 1
    )
    frame2_patches_reshaped = frame2_patches_flat.reshape(
        batch_size, H_pred, W_pred, 1, patch_size, patch_size, 1
    )

    # Squeeze the extra dimension introduced by the slice_sizes batch dimension (size 1)
    frame1_patches = jnp.squeeze(
        frame1_patches_reshaped, axis=3
    )  # Shape: (batch, H_pred, W_pred, patch_size, patch_size, 1)
    frame2_patches = jnp.squeeze(
        frame2_patches_reshaped, axis=3
    )  # Shape: (batch, H_pred, W_pred, patch_size, patch_size, 1)

    # 9. No explicit reshape needed after the final squeeze.

    # 10. Compute Photometric Loss per patch
    if loss_type == "l1":
        patch_losses = jnp.abs(frame1_patches - frame2_patches)
    elif loss_type == "l2":
        patch_losses = jnp.square(frame1_patches - frame2_patches)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # 11. Aggregate loss
    loss_per_location = jnp.mean(
        patch_losses, axis=(-3, -2, -1)
    )  # Mean over patch dims
    print(f"loss_per_location: {loss_per_location}")
    total_loss = jnp.mean(loss_per_location)  # Mean over spatial and batch dims

    return total_loss


# Example usage combined with previous modules (conceptual)
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # Simulate original images (e.g., 64x64 grayscale batch size 2)
    image1_L0 = jax.random.normal(key, (2, 64, 64, 1))
    image2_L0 = jax.random.normal(
        key, (2, 64, 64, 1)
    )  # Imagine this is shifted version of image1_L0

    # Initialize the pyramid (params shared across levels)
    pyramid_key, key = jax.random.split(key)
    pyramid_model = BaselinePyramid()
    pyramid_params = pyramid_model.init(pyramid_key, image1_L0)["params"]

    # Run Frame 1 through the pyramid
    frame1_features = pyramid_model.apply({"params": pyramid_params}, image1_L0)
    # Run Frame 2 through the pyramid
    frame2_features = pyramid_model.apply({"params": pyramid_params}, image2_L0)

    # Choose a level for prediction (e.g., Level 3, which gives 4x4 output for 64x64 input)
    L_pred = 5

    if len(frame1_features) <= L_pred:
        print(
            f"Pyramid only generated {len(frame1_features)} levels. Adjusting L_pred."
        )
        L_pred = len(frame1_features) - 1  # Use the coarsest level
        if L_pred < 0:
            raise ValueError("Pyramid generated no levels.")

    F1_Lpred = frame1_features[L_pred]  # Features for Frame 1 at L_pred
    F2_Lpred = frame2_features[L_pred]  # Features for Frame 2 at L_pred

    # Prepare input for the predictor: concatenate F1 and F2 features
    predictor_input = jnp.concatenate(
        [F1_Lpred, F2_Lpred], axis=-1
    )  # (batch, H_pred, W_pred, 8)
    H_pred, W_pred = F1_Lpred.shape[1:3]  # Get spatial dims for printing

    # Initialize the predictor
    predictor_key, key = jax.random.split(key)
    predictor_model = MinimalPredictor(hidden_features=16)
    predictor_params = predictor_model.init(predictor_key, predictor_input)["params"]

    # Predict flow at L_pred
    predicted_flow_Lpred = predictor_model.apply(
        {"params": predictor_params}, predictor_input
    )  # (batch, H_pred, W_pred, 2)

    print(
        f"\nFlow predicted at Level {L_pred} with shape: {predicted_flow_Lpred.shape}"
    )
    print(f"image1_L0 shape: {image1_L0.shape}")
    print(f"image2_L0 shape: {image2_L0.shape}")
    print(f"patch size: {3}")  # Using the hardcoded value for clarity

    # Compute the loss using the predicted flow and original images (Level 0)
    loss_value = compute_photometric_loss(
        predicted_flow_Lpred,
        image1_L0,
        image2_L0,
        L_pred=L_pred,
        patch_size=3,  # Using 3x3 patches
        loss_type="l1",  # Using L1 loss
    )

    print(f"Computed photometric loss (L1, 3x3 patches): {loss_value}")
