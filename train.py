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
NUM_EPOCHS = 5
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
    pyramid1, pyramid2, flow_pyramid, loss = model(f1=f1, f2=f2,
                                                   priors=priors, train=True)
    return loss

@nnx.jit
def train_step(model: ImagePairFlowPredictor, optimizer,
               f1: jax.Array, f2: jax.Array, priors: jax.Array):
    """
    Performs a single training step (forward pass, loss calculation, gradient computation, and parameter update).

    Args:
        state: The nnx.optim.Optimizer instance, which holds the model and optimizer state.
        batch_f1: Batch of first frames (B, 1, H, W).
        batch_f2: Batch of second frames (B, 1, H, W).

    Returns:
        - The scalar loss value for the current batch.
        - The updated nnx.Optimizer state.
    """


    loss, grads = nnx.value_and_grad(loss_fn)(model, f1, f2, priors)
    optimizer.update(grads)
    return loss

def main():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, current_time)
    writer = SummaryWriter(log_path)
    print("Initializing model and optimizer...")
    rngs = nnx.Rngs(0)
    model = ImagePairFlowPredictor(
        patch_size=PATCH_SIZE,
        channels=CHANNELS,
        levels=LEVELS,
        wavelet=WAVELET,
        ncc_patch_size=NCC_PATCH_SIZE,
        rngs=rngs,
    )
    opt = nnx.Optimizer(model, optax.adam(learning_rate=LEARNING_RATE))

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
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_data_iterator = train_dataset.as_numpy_iterator()

        for batch_idx, (f1_np, f2_np) in enumerate(train_data_iterator):
            f1_jax = jnp.array(f1_np).transpose(0, 3, 1, 2)
            f2_jax = jnp.array(f2_np).transpose(0, 3, 1, 2)
            priors = generate_random_priors(f1_jax.shape, LEVELS, PATCH_SIZE, rngs)
            loss_value = train_step(model, opt, f1_jax, f2_jax, priors)
            if global_step % 10 == 0:
                writer.add_scalar("train_loss", loss_value, global_step)
                print(f"Step {global_step}, Loss: {loss_value:.4f}")
            global_step += 1
    print("Training finished.")
    writer.close()
    print("TensorBoard writer closed.")


def generate_random_priors(input_shape,  levels, patch_size, rngs):
    B, _, H, W = input_shape
    PH = H // (2 ** levels) // patch_size
    PW = W // (2 ** levels) // patch_size

    initial_priors = jax.random.normal(rngs.prior(), (B, 2, PH, PW)) * 0.01
    return initial_priors

if __name__ == "__main__":
    main()