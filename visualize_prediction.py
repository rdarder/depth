import os
import random

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Assuming these modules are in the same project structure
from datasets import load_and_pair_frames
from predictor import MinimalPredictor
from pyramid import BaselinePyramid

# --- Configuration ---
# Match these with the values used for training in train.py
LOSS_TYPE = "l1"
CHECKPOINT_PATH = "checkpoints/model_params.npz"
TRAIN_DATA_DIR = "datasets/frames"  # Directory where your actual processed data is
IMAGE_H = 64
IMAGE_W = 64
PREDICTION_LEVEL_INDEX = (
    2  # Predict flow based on features at pyramid list index 2 (Level 3)
)
PATCH_SIZE_FOR_LOSS = 3  # Use 3x3 patches for the loss calculation
PREDICTOR_HIDDEN_FEATURES = 16  # Size of the hidden layer in the predictor
# --- End Configuration ---

# Image dimensions
IMAGE_SIZE = (IMAGE_H, IMAGE_W)  # HxW
IMAGE_SHAPE_L0 = (1, IMAGE_H, IMAGE_W, 1)  # (batch=1, H, W, C=1)

# Global flag to signal exit from the visualization loop
exit_visualization = False


# Callback function for keyboard events
def on_key_press(event):
    """Callback function for keyboard press events in the plot window."""
    global exit_visualization
    if event.key == "q":
        print("Exiting visualization loop...")
        exit_visualization = True
    # Close the current figure on any key press to advance the loop
    plt.close(event.canvas.figure)


def load_model_params(params_path: str):
    """Loads model parameters from a .npz file."""
    if not os.path.exists(params_path):
        print(f"Error: Model parameters file not found at {params_path}")
        return None
    try:
        # Load the numpy arrays
        params_numpy = np.load(params_path, allow_pickle=True)
        # Load the nested dictionary saved under the 'params' key
        params_dict = params_numpy[
            "params"
        ].item()  # Use .item() to get the dictionary from the 0-dim array
        print(f"Successfully loaded parameters from {params_path}")
        return params_dict
    except Exception as e:
        print(f"Error loading parameters from {params_path}: {e}")
        return None


def initialize_models(
    pyramid_model: nn.Module,
    predictor_model: nn.Module,
    dummy_image_shape: tuple,
    L_pred_index: int,
    hidden_features_predictor: int,
    key: jax.random.PRNGKey,
):
    """Initializes model structures with dummy data to get the parameter tree."""
    pyramid_key, predictor_key, key = jax.random.split(key, 3)

    # Initialize pyramid structure
    dummy_image = jnp.ones(dummy_image_shape, dtype=jnp.float32)
    pyramid_variables = pyramid_model.init(pyramid_key, dummy_image)
    dummy_pyramid_features = pyramid_model.apply(pyramid_variables, dummy_image)

    if L_pred_index >= len(dummy_pyramid_features):
        raise ValueError(
            f"L_pred_index ({L_pred_index}) out of bounds for pyramid output "
            f"(generated {len(dummy_pyramid_features)} levels)."
        )

    dummy_F_Lpred = dummy_pyramid_features[L_pred_index]
    dummy_predictor_input_shape = dummy_F_Lpred.shape[:-1] + (8,)

    # Initialize predictor structure
    dummy_predictor_input = jnp.ones(dummy_predictor_input_shape, dtype=jnp.float32)
    predictor_variables = predictor_model.init(predictor_key, dummy_predictor_input)

    # Return dummy parameters in the correct structure
    dummy_params = {
        "pyramid": pyramid_variables["params"],
        "predictor": predictor_variables["params"],
    }
    return dummy_params


def predict_flow(
    params,
    image1_L0: jax.Array,  # (1, H, W, 1)
    image2_L0: jax.Array,  # (1, H, W, 1)
    pyramid_model: nn.Module,
    predictor_model: nn.Module,
    L_pred_index: int,
):
    """Performs forward pass to predict flow for a single frame pair."""
    # Ensure inputs have a batch dimension (size 1)
    image1_L0 = jnp.expand_dims(image1_L0, axis=0) if image1_L0.ndim == 3 else image1_L0
    image2_L0 = jnp.expand_dims(image2_L0, axis=0) if image2_L0.ndim == 3 else image2_L0

    # Run images through the pyramid model
    frame1_features = pyramid_model.apply({"params": params["pyramid"]}, image1_L0)
    frame2_features = pyramid_model.apply({"params": params["pyramid"]}, image2_L0)

    # Select features for the chosen prediction level
    F1_Lpred = frame1_features[L_pred_index]
    F2_Lpred = frame2_features[L_pred_index]

    # Prepare input for the predictor
    predictor_input = jnp.concatenate([F1_Lpred, F2_Lpred], axis=-1)

    # Run features through the predictor model
    predicted_flow_Lpred = predictor_model.apply(
        {"params": params["predictor"]}, predictor_input
    )

    # Predicted flow has shape (1, H_pred, W_pred, 2)
    # Squeeze the batch dimension for easier handling later
    return jnp.squeeze(predicted_flow_Lpred, axis=0)  # Shape (H_pred, W_pred, 2)


def get_patch_from_image(image, top_left_i, top_left_j, patch_size):
    """Extracts a single patch from an image."""
    # Image shape is (H, W, 1), start_indices is (i, j, c)
    start_indices = (top_left_i, top_left_j, 0)
    slice_sizes = (patch_size, patch_size, 1)
    return jax.lax.dynamic_slice(image, start_indices, slice_sizes)


def main_visualization_loop():
    print("\n--- Visualization Setup ---")

    # Initialize models to get the structure
    key = jax.random.PRNGKey(0)  # Using a fixed key for initialization structure
    pyramid_model = BaselinePyramid()
    predictor_model = MinimalPredictor(hidden_features=PREDICTOR_HIDDEN_FEATURES)

    # Need a dummy image shape for initialization. Batch size doesn't matter here.
    dummy_image_shape_init = (1, IMAGE_H, IMAGE_W, 1)
    try:
        dummy_params = initialize_models(
            pyramid_model,
            predictor_model,
            dummy_image_shape_init,
            PREDICTION_LEVEL_INDEX,
            PREDICTOR_HIDDEN_FEATURES,
            key,
        )
    except ValueError as e:
        print(f"Error during model initialization: {e}. Exiting.")
        return

    # Load trained parameters
    loaded_params_dict = load_model_params(CHECKPOINT_PATH)
    if loaded_params_dict is None:
        return

    # Restore the loaded parameters into the model structure
    # We need to use the structure from dummy_params
    try:
        restored_params = flax.serialization.from_state_dict(
            dummy_params, loaded_params_dict
        )
        print("Model parameters restored successfully.")
    except Exception as e:
        print(
            f"Error restoring parameters: {e}. Ensure the model architecture "
            f"and saved parameters match. Exiting."
        )
        return

    print(f"Predicting flow at pyramid level index: {PREDICTION_LEVEL_INDEX}")
    print(f"Using patch size: {PATCH_SIZE_FOR_LOSS}")
    print("-" * 20)

    # Determine H_pred, W_pred from the shape of dummy features at L_pred_index
    # This logic is similar to train.py
    dummy_image_batch = jnp.ones((1, IMAGE_H, IMAGE_W, 1), dtype=jnp.float32)
    dummy_pyramid_variables = pyramid_model.init(key, dummy_image_batch)
    dummy_pyramid_features = pyramid_model.apply(
        dummy_pyramid_variables, dummy_image_batch
    )
    H_pred, W_pred = dummy_pyramid_features[PREDICTION_LEVEL_INDEX].shape[1:3]
    scale = 2**PREDICTION_LEVEL_INDEX
    patch_half_size = PATCH_SIZE_FOR_LOSS // 2

    print(f"Flow prediction grid size (H_pred, W_pred): ({H_pred}, {W_pred})")

    # Load dataset
    dataset = load_and_pair_frames(data_dir=TRAIN_DATA_DIR, image_size=IMAGE_SIZE)
    if dataset is None or dataset.shape[0] == 0:
        print("Failed to load dataset or dataset is empty. Exiting.")
        return

    total_pairs = dataset.shape[0]
    print(f"Successfully loaded {total_pairs} frame pairs for visualization.")

    while True:
        # 1. Select a random frame pair
        random_index = random.randint(0, total_pairs - 1)
        frame_pair = dataset[random_index]  # Shape (2, H, W, 1)
        image1_L0 = frame_pair[0]  # Shape (H, W, 1)
        image2_L0 = frame_pair[1]  # Shape (H, W, 1)

        # 2. Predict flow for this pair
        # predict_flow expects (1, H, W, 1), so we pass the unsqueezed images
        predicted_flow_Lpred = predict_flow(
            restored_params,
            image1_L0,
            image2_L0,
            pyramid_model,
            predictor_model,
            PREDICTION_LEVEL_INDEX,
        )  # Shape (H_pred, W_pred, 2)

        # 3. Choose a random location (pixel) in the predicted flow grid
        random_pred_i = random.randint(0, H_pred - 1)
        random_pred_j = random.randint(0, W_pred - 1)

        # Get the predicted flow vector at this location
        flow_vector_pred = predicted_flow_Lpred[random_pred_i, random_pred_j]

        # 4. Calculate the corresponding patch center and top-left coordinates in L0
        # Center in L0 for Frame 1
        center1_i_L0 = random_pred_i * scale
        center1_j_L0 = random_pred_j * scale

        # Predicted center in L0 for Frame 2
        # Flow vector (dx, dy) here represents displacement in the L0 grid
        # Based on compute_photometric_loss, flow[..., 0] is vertical (i) and flow[..., 1] is horizontal (j)
        predicted_center2_i_L0 = center1_i_L0 + flow_vector_pred[0]
        predicted_center2_j_L0 = center1_j_L0 + flow_vector_pred[1]

        # Calculate top-left coordinates for the patch in Frame 1 (L0)
        start1_i_L0_raw = round(center1_i_L0 - patch_half_size)
        start1_j_L0_raw = round(center1_j_L0 - patch_half_size)

        # Calculate predicted top-left coordinates for the patch in Frame 2 (L0)
        start2_i_L0_raw = round(predicted_center2_i_L0 - patch_half_size)
        start2_j_L0_raw = round(predicted_center2_j_L0 - patch_half_size)

        # Clamp coordinates to ensure the patch is within image bounds
        start1_i_L0_clamped = int(
            np.clip(start1_i_L0_raw, 0, IMAGE_H - PATCH_SIZE_FOR_LOSS)
        )
        start1_j_L0_clamped = int(
            np.clip(start1_j_L0_raw, 0, IMAGE_W - PATCH_SIZE_FOR_LOSS)
        )

        start2_i_L0_clamped = int(
            np.clip(start2_i_L0_raw, 0, IMAGE_H - PATCH_SIZE_FOR_LOSS)
        )
        start2_j_L0_clamped = int(
            np.clip(start2_j_L0_raw, 0, IMAGE_W - PATCH_SIZE_FOR_LOSS)
        )

        # 5. Extract patches from the original images
        patch1 = get_patch_from_image(
            image1_L0, start1_i_L0_clamped, start1_j_L0_clamped, PATCH_SIZE_FOR_LOSS
        )
        patch2 = get_patch_from_image(
            image2_L0, start2_i_L0_clamped, start2_j_L0_clamped, PATCH_SIZE_FOR_LOSS
        )

        # 6. Calculate loss for the extracted patches and determine common vmin/vmax for plotting
        # Compute the raw L1/L2 difference on the patches and average it.

        # Determine the overall min/max across both patches for consistent visualization
        patch_min = jnp.min(jnp.stack([patch1, patch2]))
        patch_max = jnp.max(jnp.stack([patch1, patch2]))

        # Compute the raw L1/L2 difference on the patches and average it.
        # Need to add a batch dimension of 1 for the loss function which expects (batch, ...)\
        # Note: We calculate the loss directly on the patches here for display purposes,
        # matching the core patch comparison logic within compute_photometric_loss.
        # However, compute_photometric_loss is designed for the whole batch.
        # A simpler way is to just compute the raw L1/L2 difference on the patches
        # and average it, matching the patch part of compute_photometric_loss.
        if LOSS_TYPE == "l1":
            patch_loss_value = jnp.mean(jnp.abs(patch1 - patch2))
        elif LOSS_TYPE == "l2":
            patch_loss_value = jnp.mean(jnp.square(patch1 - patch2))
        else:
            # Should match loss_type used in training, but handle unexpected value
            print(f"Warning: Unsupported loss type '{LOSS_TYPE}' for display.")
            patch_loss_value = -1  # Indicate calculation issue

        # 7. Visualize
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Plot full images with patches highlighted
        # Ensure consistent color scaling for the full images
        axes[0, 0].imshow(image1_L0.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Frame 1 with Patch")
        rect1 = patches.Rectangle(
            (start1_j_L0_clamped, start1_i_L0_clamped),  # (x, y) -> (col, row)
            PATCH_SIZE_FOR_LOSS,
            PATCH_SIZE_FOR_LOSS,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        axes[0, 0].add_patch(rect1)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(image2_L0.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title(
            f"Frame 2 with flow. Loss ({LOSS_TYPE}): {patch_loss_value:.6f}"
        )

        # Mark original patch location in Frame 2 (red)
        # Coordinates are the same as Frame 1 patch location
        rect2_original = patches.Rectangle(
            (start1_j_L0_clamped, start1_i_L0_clamped),  # (x, y) -> (col, row)
            PATCH_SIZE_FOR_LOSS,
            PATCH_SIZE_FOR_LOSS,
            linewidth=1,
            edgecolor="red",
            facecolor="none",
            label="Original Location",  # Add label for legend
        )
        axes[0, 1].add_patch(rect2_original)

        # Mark predicted patch location in Frame 2 (green)
        rect2_predicted = patches.Rectangle(
            (start2_j_L0_clamped, start2_i_L0_clamped),  # (x, y) -> (col, row)
            PATCH_SIZE_FOR_LOSS,
            PATCH_SIZE_FOR_LOSS,
            linewidth=1,
            edgecolor="green",  # Changed to green
            facecolor="none",
            label="Predicted Location",  # Add label for legend
        )
        axes[0, 1].add_patch(rect2_predicted)

        # Add arrow for predicted flow vector
        # Start from the center of the original location, end at the center of the predicted location
        # Matplotlib arrow(x, y, dx, dy)
        arrow_start_x = center1_j_L0  # Column
        arrow_start_y = center1_i_L0  # Row
        arrow_end_x = predicted_center2_j_L0
        arrow_end_y = predicted_center2_i_L0
        arrow_dx = arrow_end_x - arrow_start_x
        arrow_dy = arrow_end_y - arrow_start_y

        axes[0, 1].arrow(
            arrow_start_x,
            arrow_start_y,
            arrow_dx,
            arrow_dy,
            head_width=max(
                1, PATCH_SIZE_FOR_LOSS / 4
            ),  # Make arrow head size reasonable
            head_length=max(1, PATCH_SIZE_FOR_LOSS / 4),
            fc="cyan",
            ec="cyan",
            length_includes_head=True,
        )

        # Add text annotation for flow vector (dx, dy)
        # Position text near the start of the arrow or patch
        axes[0, 1].text(
            arrow_start_x,
            arrow_start_y - patch_half_size - 2,  # Position slightly above the patch
            f"({flow_vector_pred[1]:.2f}, {flow_vector_pred[0]:.2f})",  # (dx, dy) -> (j, i)
            color="cyan",
            fontsize=8,
            ha="center",
            va="bottom",
        )

        # Add legend to frame 2 plot
        axes[0, 1].legend(loc="upper right")

        axes[0, 1].axis("off")

        # Plot extracted patches
        # Ensure consistent color scaling for the patches
        axes[1, 0].imshow(patch1.squeeze(), cmap="gray", vmin=patch_min, vmax=patch_max)
        axes[1, 0].set_title("Predicted path at frame 1")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(patch2.squeeze(), cmap="gray", vmin=patch_min, vmax=patch_max)
        axes[1, 1].set_title("Predicted patch at frame 2")
        axes[1, 1].axis("off")

        # Add text annotation for the patch loss
        # fig.suptitle(
        #     f"Patch Loss ({LOSS_TYPE}): {patch_loss_value:.6f}", fontsize=12, y=1.02
        # )  # Add title above plots

        fig.tight_layout(
            rect=[0, 0, 1, 0.98]
        )  # Adjust layout to make space for suptitle

        fig.canvas.mpl_connect("key_press_event", on_key_press)

        plt.show()

        if exit_visualization:
            break


if __name__ == "__main__":
    main_visualization_loop()
