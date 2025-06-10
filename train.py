from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from datetime import datetime

from tensorboardX import SummaryWriter
import os
import orbax.checkpoint

from datasets import FrameSource, FramePairsDataset
from depth import ImagePairFlowPredictor

TENSORBOARD_LOGS = "./runs"
CHECKPOINTS_DIR = Path("./checkpoints")

TRAIN_DATASET_ROOT = Path('datasets/frames')
MAX_FRAMES_LOOKAHEAD = 5

LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 50

PATCH_SIZE = 2
NCC_PATCH_SIZE = 4
CHANNELS = 4
LEVELS = 6


def loss_fn(model, f1, f2, priors):
    pyramid1, pyramid2, flow_pyramid, loss, aux_data, per_patch_loss_maps = model(
        f1=f1, f2=f2, priors=priors, train=True
    )
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
        - The auxiliary data dictionary.
        - The updated nnx.Optimizer (implicitly returned as nnx updates state in place).
    """
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, f1, f2, priors)
    optimizer.update(grads)
    return loss, aux


def main():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # --- Use same timestamp for logs and checkpoints for easy correlation ---
    log_path = os.path.join(TENSORBOARD_LOGS, current_time)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)  # Ensure checkpoint directory exists
    checkpoint_dir = CHECKPOINTS_DIR / current_time
    # --- END CHANGE ---

    writer = SummaryWriter(log_path)
    print("Initializing model and optimizer...")
    rngs = nnx.Rngs(params=0, prior=jax.random.PRNGKey(0))

    model = ImagePairFlowPredictor(
        patch_size=PATCH_SIZE,
        channels=CHANNELS,
        levels=LEVELS,
        ncc_patch_size=NCC_PATCH_SIZE,
        rngs=rngs,
        loss_alpha=2.0,
        loss_beta=1.0,
        loss_gamma=1.0
    )

    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=LEARNING_RATE)
    ))

    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()

    print("loading dataset")
    source = FrameSource(Path('datasets/frames').glob('*'), cache_size=10_000)
    train_dataset = FramePairsDataset(
        source,
        batch_size=BATCH_SIZE,
        max_frame_lookahead=MAX_FRAMES_LOOKAHEAD,
        include_reversed_pairs=True,
        drop_uneven_batches=True,
        seed=0
    )

    print("Starting training loop...")
    print(f"{len(train_dataset)} frame pairs on {BATCH_SIZE} size batches over {NUM_EPOCHS} epochs")
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"epoch {epoch}")
        for step, (f1_jax, f2_jax) in enumerate(train_dataset):

            priors = generate_random_priors(f1_jax.shape, LEVELS, PATCH_SIZE, rngs)
            loss_value, aux_data = train_step(model, optimizer, f1_jax, f2_jax, priors)

            if not jnp.isfinite(loss_value):
                print(f"Warning: NaN or Inf loss detected at step {step}. Exiting training.")
                break

            if global_step % 10 == 0:
                writer.add_scalar("train_loss", loss_value, global_step)
                print(f"Step {global_step}, Total Weighted Loss: {loss_value:.4f}")
                mean_unweighted_losses = aux_data[
                    'mean_unweighted_losses_per_level']  # (N_Levels,) array
                for level_idx, mean_unweighted_loss in enumerate(mean_unweighted_losses):
                    # Use clear, consistent logging tags
                    writer.add_scalar(f"train_loss_levels/{level_idx}", mean_unweighted_loss,
                                      global_step)
            global_step += 1

    print("Training finished.")
    writer.close()

    model_state = nnx.state(model)
    orbax_checkpointer.save(checkpoint_dir.absolute() / 'final', model_state)
    orbax_checkpointer.wait_until_finished()
    print("Checkpoint saving complete.")


def generate_random_priors(input_shape, levels, patch_size, rngs: nnx.Rngs):
    B, _, H, W = input_shape
    PH = H // (2 ** levels) // patch_size
    PW = W // (2 ** levels) // patch_size

    initial_priors = jax.random.normal(rngs.prior(), (B, 2, PH, PW)) * 0.01
    return initial_priors


if __name__ == "__main__":
    main()

# --- END OF FILE train.py ---
