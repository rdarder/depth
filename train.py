from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from datetime import datetime

from flax.training import train_state
from tensorboardX import SummaryWriter  # For TensorBoard logging
import os
import numpy as np  # For converting tf.data.Dataset outputs to numpy arrays

# Import your custom modules
from depth import ImagePairFlowPredictor  # Your model
import datasets as custom_datasets  # Your dataset loading utility

# --- Training Configuration ---
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
LOG_DIR = "./runs"  # TensorBoard logs will be saved here
# MODEL_DIR = "./checkpoints" # Optional: Directory for saving model checkpoints

# Dataset specific configurations (from datasets.py, can be overridden here)
DATASET_ROOT = custom_datasets.DATASET_DIR
IMAGE_EXTENSION = custom_datasets.IMAGE_EXTENSION
TARGET_IMG_HEIGHT = custom_datasets.TARGET_IMG_HEIGHT
TARGET_IMG_WIDTH = custom_datasets.TARGET_IMG_WIDTH
SHUFFLE_BUFFER_SIZE = custom_datasets.SHUFFLE_BUFFER_SIZE
BATCH_SIZE = 100

# Model specific configurations (from depth.py)
PATCH_SIZE = 2  # Patch size for flow estimation (within DWT levels)
CHANNELS = 4  # Number of channels from DWT output
LEVELS = 5  # Number of DWT levels
WAVELET = 'db2'
NCC_PATCH_SIZE = 4  # Patch size for photometric loss (on finer/original resolution)


def loss_fn(model, f1, f2, priors):
    # model now returns (pyramid1, pyramid2, flow_pyramid, loss, weights)
    pyramid1, pyramid2, flow_pyramid, loss, aux_data = model(
        f1=f1, f2=f2, priors=priors, train=True
    )
    # Return loss and weights (auxiliary data)
    return loss, aux_data


@nnx.jit
def train_step(model: ImagePairFlowPredictor, optimizer: nnx.Optimizer,
               f1: jax.Array, f2: jax.Array, priors: jax.Array):
    """
    Performs a single training step (forward pass, loss calculation, gradient computation, and parameter update).

    Args:
        model: The nnx.Module instance (ImagePairFlowPredictor).
        optimizer: The nnx.Optimizer instance.
        f1: Batch of first frames (B, 1, H, W).
        f2: Batch of second frames (B, 1, H, W).
        priors: Initial flow priors for the coarsest level (B, 2, H_coarsest_patches, W_coarsest_patches).

    Returns:
        - The scalar loss value for the current batch.
        - The auxiliary data (list of weights).
        - The updated nnx.Optimizer (implicitly returned as nnx updates state in place).
    """
    # Use nnx.value_and_grad with has_aux=True
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, f1, f2, priors)
    optimizer.update(grads)  # nnx.Optimizer updates the model in-place
    # Return loss and auxiliary data
    return loss, aux


# --- END CHANGE ---


def main():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, current_time)
    writer = SummaryWriter(log_path)
    print("Initializing model and optimizer...")
    rngs = nnx.Rngs(params=0, prior=jax.random.PRNGKey(0))

    # --- CHANGE: Pass loss hyperparameters to model init ---
    # Use default values for now, these can be added to config later
    model = ImagePairFlowPredictor(
        patch_size=PATCH_SIZE,
        channels=CHANNELS,
        levels=LEVELS,
        wavelet=WAVELET,
        ncc_patch_size=NCC_PATCH_SIZE,
        rngs=rngs,
        loss_alpha=5.0,  # Default alpha
        loss_beta=1.0,  # Default beta
        loss_gamma=1.0  # Default gamma for coarsest level
    )
    # --- END CHANGE ---

    # Add gradient clipping (as previously discussed and implemented)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients by global norm of 1.0
        optax.adam(learning_rate=LEARNING_RATE)
    ))

    print("loading dataset")
    frame_pairs = custom_datasets.get_consecutive_frame_pairs(DATASET_ROOT, IMAGE_EXTENSION)
    train_dataset = custom_datasets.create_dataset(
        frame_pairs,
        target_height=TARGET_IMG_HEIGHT,
        target_width=TARGET_IMG_WIDTH,
        batch_size=BATCH_SIZE,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        is_training=True,
    )

    print("Starting training loop...")
    global_step = 0
    # --- CHANGE: Iterate over dataset directly (it's iterable) ---
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        # --- CHANGE: Handle auxiliary output from train_step ---
        for batch_idx, (f1_np, f2_np) in enumerate(train_dataset):
            f1_jax = jnp.array(f1_np).transpose(0, 3, 1, 2)  # From (B, H, W, C) to (B, C, H, W)
            f2_jax = jnp.array(f2_np).transpose(0, 3, 1, 2)

            priors = generate_random_priors(f1_jax.shape, LEVELS, PATCH_SIZE, rngs)

            # train_step returns (loss, weights)
            loss_value, aux_data = train_step(model, optimizer, f1_jax, f2_jax, priors)

            # Check for NaN/Inf loss (optional, but good for debugging)
            if not jnp.isfinite(loss_value):
                print(f"Warning: NaN or Inf loss detected at step {global_step}. Exiting training.")
                break

                # Log loss and weights
            if global_step % 10 == 0:
                writer.add_scalar("train_loss", loss_value, global_step)
                print(f"Step {global_step}, Loss: {loss_value:.4f}")

                weights = aux_data['weights']  # List of (B,) arrays
                for level_idx, level_weights_batch in enumerate(weights):
                    mean_weight = jnp.mean(level_weights_batch)  # Mean across the batch
                    writer.add_scalar(f"level_train_weight/{level_idx}", mean_weight,
                                      global_step)

                # Log average unweighted loss per level
                mean_unweighted_losses = aux_data[
                    'mean_unweighted_losses_per_level']  # (N_Levels,) array
                for level_idx, mean_unweighted_loss in enumerate(mean_unweighted_losses):
                    writer.add_scalar(f"unweighted_train_loss_level/{level_idx}",
                                      mean_unweighted_loss, global_step)

            global_step += 1
        # Check for NaN/Inf after inner loop
        if not jnp.isfinite(loss_value):
            break
        # --- END CHANGE ---

    print("Training finished.")
    writer.close()
    print("TensorBoard writer closed.")


def generate_random_priors(input_shape, levels, patch_size, rngs):
    B, _, H, W = input_shape
    PH = H // (2 ** levels) // patch_size
    PW = W // (2 ** levels) // patch_size

    initial_priors = jax.random.normal(rngs.prior(), (B, 2, PH, PW)) * 0.01
    return initial_priors


if __name__ == "__main__":
    main()
