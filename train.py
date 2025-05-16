import os  # Import os for directory creation
from functools import partial

import flax.linen as nn
import flax.serialization  # Import serialization module
import jax
import jax.numpy as jnp
import numpy as np  # Import numpy for saving
import optax
from jax import jit

from datasets import (
    PROCESSED_DATA_DIR,
    TARGET_IMAGE_SIZE,
    create_batches,
    load_and_pair_frames,
)
from loss import compute_photometric_loss
from predictor import MinimalPredictor
from pyramid import BaselinePyramid


def initialize_models_and_optimizer(
    pyramid_model: nn.Module,
    predictor_model: nn.Module,
    dummy_image_shape: tuple,  # (batch_size, H, W, 1)
    predictor_input_shape: tuple,  # (batch_size, H_pred, W_pred, 8)
    learning_rate: float,
    key: jax.random.PRNGKey,
):
    """Initializes model parameters and the optax optimizer."""
    pyramid_key, predictor_key, params_key = jax.random.split(
        key, 3
    )  # Need key for params init

    # Initialize pyramid parameters
    # We only need parameters for the shared conv_level
    pyramid_params = pyramid_model.init(pyramid_key, jnp.ones(dummy_image_shape))[
        "params"
    ]

    # Initialize predictor parameters
    # We need parameters for the Dense layers
    predictor_params = predictor_model.init(
        predictor_key, jnp.ones(predictor_input_shape)
    )["params"]

    # Combine parameters into a single dictionary
    # optax works best with a single nested parameter tree
    all_params = {"pyramid": pyramid_params, "predictor": predictor_params}

    # Define the optimizer
    optimizer_def = optax.adam(learning_rate)

    # Initialize the optimizer state using the combined parameters
    optimizer_state = optimizer_def.init(all_params)

    return all_params, optimizer_state, optimizer_def


@partial(
    jit,
    static_argnames=[
        "optimizer_def",
        "pyramid_model",
        "predictor_model",
        "L_pred_index",
        "patch_size",
        "loss_type",
    ],
)
def train_step(
    params,  # Current model parameters
    optimizer_state,  # Current optimizer state
    optimizer_def,  # The optimizer definition
    batch: jax.Array,  # Batch of data (batch_size, 2, H, W, 1)
    pyramid_model: nn.Module,
    predictor_model: nn.Module,
    L_pred_index: int,  # Index of the pyramid level for prediction
    patch_size: int,
    loss_type: str,
):
    """Performs one training step using optax."""

    # Define the loss function for gradient computation
    # This function takes only the parameters that we want to differentiate with respect to
    def loss_fn(params_to_diff):
        # Unpack batch
        image1_L0 = batch[:, 0]  # (batch_size, H, W, 1)
        image2_L0 = batch[:, 1]  # (batch_size, H, W, 1)

        # 1. Run images through the pyramid model
        # Apply requires the parameters nested under 'params'
        frame1_features = pyramid_model.apply(
            {"params": params_to_diff["pyramid"]}, image1_L0
        )
        frame2_features = pyramid_model.apply(
            {"params": params_to_diff["pyramid"]}, image2_L0
        )

        # Select features for the chosen prediction level
        # Assuming L_pred_index is valid (checked in main loop)
        F1_Lpred = frame1_features[L_pred_index]  # (batch, H_pred, W_pred, 4)
        F2_Lpred = frame2_features[L_pred_index]  # (batch, H_pred, W_pred, 4)

        # Prepare input for the predictor
        predictor_input = jnp.concatenate(
            [F1_Lpred, F2_Lpred], axis=-1
        )  # (batch, H_pred, W_pred, 8)

        # 2. Run features through the predictor model
        predicted_flow_Lpred = predictor_model.apply(
            {"params": params_to_diff["predictor"]}, predictor_input
        )  # (batch, H_pred, W_pred, 2)

        # predicted_flow_Lpred = jnp.ones_like(predicted_flow_Lpred) * 4.0

        # 3. Compute loss
        # L_pred is the level NUMBER, not the index (index + 1)
        L_pred = L_pred_index + 1
        loss = compute_photometric_loss(
            predicted_flow_Lpred,
            image1_L0,
            image2_L0,
            L_pred=L_pred,
            patch_size=patch_size,
            loss_type=loss_type,
        )

        return loss

    # Compute the loss and gradients using jax.grad
    # We differentiate with respect to the combined 'params' dictionary
    loss_value, grads = jax.value_and_grad(loss_fn)(params)

    # Compute parameter updates and get the new optimizer state
    updates, new_optimizer_state = optimizer_def.update(grads, optimizer_state)

    # Apply the updates to the parameters
    new_params = optax.apply_updates(params, updates)

    return new_params, new_optimizer_state, loss_value


def main_training_loop(
    data_dir: str = PROCESSED_DATA_DIR,
    image_size: tuple = TARGET_IMAGE_SIZE,
    batch_size: int = 8,
    learning_rate: float = 1e-4,  # Adam default learning rate
    num_epochs: int = 10,
    L_pred_index: int = 2,  # Default: Predict at level 3 (0-indexed 2)
    patch_size: int = 3,  # Default: 5x5 patches for loss
    loss_type: str = "l2",  # Default: 'l1' or 'l2'
    hidden_features_predictor: int = 16,
    seed: int = 0,
):
    """Main function to load data, train models, and report progress."""

    key = jax.random.PRNGKey(seed)

    # 1. Load Dataset
    key, load_key = jax.random.split(key)
    dataset = load_and_pair_frames(data_dir=data_dir, image_size=image_size)

    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return

    total_pairs = dataset.shape[0]
    H_L0, W_L0 = dataset.shape[2:4]  # Original image dimensions

    # Determine shape for predictor input based on the chosen prediction level
    # Need to run a dummy image through the pyramid once to find the shape
    dummy_image_batch = jnp.ones((batch_size, H_L0, W_L0, 1), dtype=jnp.float32)
    pyramid_model_dummy = BaselinePyramid()
    # Use a dummy key for apply shape inference, actual init uses real key later
    dummy_key = jax.random.PRNGKey(0)  # Using a static key for shape inference
    dummy_pyramid_variables = pyramid_model_dummy.init(dummy_key, dummy_image_batch)
    dummy_pyramid_features = pyramid_model_dummy.apply(
        dummy_pyramid_variables, dummy_image_batch
    )

    if L_pred_index >= len(dummy_pyramid_features):
        print(
            f"Error: L_pred_index ({L_pred_index}) is out of bounds for pyramid output (generated {len(dummy_pyramid_features)} levels)."
        )
        print("Please choose a smaller L_pred_index.")
        return

    dummy_F_Lpred = dummy_pyramid_features[L_pred_index]
    dummy_predictor_input_shape = dummy_F_Lpred.shape[:-1] + (
        8,
    )  # (batch_size, H_pred, W_pred, 8)

    print("\n--- Training Setup ---")
    print(
        f"Predicting flow at pyramid level index: {L_pred_index} (Level Number: {L_pred_index + 1})"
    )
    print(f"Features for prediction have shape: {dummy_F_Lpred.shape}")
    print(f"Predictor input shape: {dummy_predictor_input_shape}")
    print(
        f"Photometric loss using {patch_size}x{patch_size} patches on Level 0 images."
    )
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Total data pairs: {total_pairs}")
    print("-" * 20)

    # 2. Initialize Models and Optimizer (using optax)
    key, init_key = jax.random.split(key)
    initial_params, optimizer_state, optimizer_def = initialize_models_and_optimizer(
        BaselinePyramid(),  # Pass the class
        MinimalPredictor(
            hidden_features=hidden_features_predictor
        ),  # Pass instance with hparams
        dummy_image_batch.shape,  # Shape needed for pyramid init
        dummy_predictor_input_shape,  # Shape needed for predictor init
        learning_rate,
        init_key,
    )
    print("Models and Optax Optimizer initialized.")

    # Initialize trainable parameters
    current_params = initial_params

    # 3. Training Loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        key, shuffle_key = jax.random.split(key)  # Split key for shuffling each epoch

        total_epoch_loss = 0
        num_batches = 0

        # Use create_batches generator
        for batch in create_batches(dataset, batch_size, shuffle=True, key=shuffle_key):
            # Perform one training step
            # train_step returns the *new* parameters and optimizer state
            current_params, optimizer_state, loss_value = train_step(
                current_params,
                optimizer_state,
                optimizer_def,  # Pass the optimizer definition
                batch,
                BaselinePyramid(),  # Pass class for JIT
                MinimalPredictor(
                    hidden_features=hidden_features_predictor
                ),  # Pass instance for JIT
                L_pred_index,
                patch_size,
                loss_type,
            )
            total_epoch_loss += loss_value
            num_batches += 1

        avg_epoch_loss = total_epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")

    print("Training finished.")
    # The final learned parameters are in `current_params`

    # --- Save the final learned parameters ---
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    params_path = os.path.join(checkpoint_dir, "model_params.npz")

    # Convert parameters from JAX arrays to NumPy arrays for saving
    # Using to_state_dict helps standardize the format
    params_state = flax.serialization.to_state_dict(current_params)
    # Convert all JAX arrays in the state dict to numpy arrays
    params_numpy = jax.tree_util.tree_map(np.asarray, params_state)

    # Save the nested dictionary under a single key
    np.savez(params_path, params=params_numpy)
    print(f"Model parameters saved to {params_path}")

    # print(\"Final parameters:\", current_params) # Will print nested dict structure


if __name__ == "__main__":
    # --- Script Configuration ---
    TRAIN_DATA_DIR = "datasets/frames"  # Directory where your actual processed data is
    IMAGE_H = 64
    IMAGE_W = 64
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10  # Start with a moderate number
    # --- Experiment Configuration ---
    PREDICTION_LEVEL_INDEX = (
        1  # Predict flow based on features at pyramid list index 2 (Level 3)
    )
    PATCH_SIZE_FOR_LOSS = 3  # Use 5x5 patches for the loss calculation
    LOSS_TYPE = "l1"  # 'l1' or 'l2'
    PREDICTOR_HIDDEN_FEATURES = 16  # Size of the hidden layer in the predictor

    # --- Run the training ---
    # Note: Make sure you have run the shell script first to populate PROCESSED_DATA_DIR
    # with 64x64 grayscale frames.
    # If you use the placeholders, the script will run but train on random data.

    main_training_loop(
        data_dir=TRAIN_DATA_DIR,
        image_size=(IMAGE_H, IMAGE_W),
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        L_pred_index=PREDICTION_LEVEL_INDEX,
        patch_size=PATCH_SIZE_FOR_LOSS,
        loss_type=LOSS_TYPE,
        hidden_features_predictor=PREDICTOR_HIDDEN_FEATURES,
        seed=42,  # Use a fixed seed for reproducibility
    )
