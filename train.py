import os
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf  # For tf.data pipeline
from flax.metrics import tensorboard

from datasets import create_dataset, get_consecutive_frame_pairs
from flow_model import OpticalFlow, loss_fn_for_grad

# --- Training Configuration ---
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 4  # Should match dataset batch size
NUM_PYRAMID_LEVELS = 3
PREDICTOR_HIDDEN_FEATURES = 32
PATCH_SIZE_LOSS = 5
LOG_EVERY_N_STEPS = 200
TENSORBOARD_LOG_DIR = "./tensorboard_logs"
MODEL_SAVE_DIR = "./saved_models"

# Dataset params (should match dataset script)
DATASET_DIR = "datasets/frames"  # or your actual dataset path
IMAGE_EXTENSION = ".png"
TARGET_IMG_HEIGHT = 64  # Example, make sure divisible for pyramid
TARGET_IMG_WIDTH = 64  # Example


def initialize_model_and_optimizer(key: jax.random.PRNGKey, learning_rate: float):
    """Initializes the model and the optimizer."""
    model_key, _ = jax.random.split(key)

    # Initialize the main model (which initializes submodules)
    # NNX modules are stateful, initialization happens at construction.
    # rngs=nnx.Rngs(params=model_key) helps if submodules need different keys based on collection.
    # For simple nnx.Param, just one key is often fine.
    model = OpticalFlow(
        num_pyramid_levels=NUM_PYRAMID_LEVELS,
        predictor_hidden_features=PREDICTOR_HIDDEN_FEATURES,
        rngs=nnx.Rngs(params=model_key),  # Pass Rngs for parameter initialization
    )

    # Create the optimizer using nnx.Optimizer
    # It will manage the parameters of the 'model' instance.
    tx = optax.adam(learning_rate=learning_rate)
    optimizer = nnx.Optimizer(model, tx)

    return model, optimizer


# @partial(nnx.jit, static_argnums=(4))  # JIT compile the training step
def train_step(
    model: OpticalFlow,
    optimizer: nnx.Optimizer,
    batch_frame1: jax.Array,
    batch_frame2: jax.Array,
    patch_size_loss: int,
):
    """Performs a single training step."""
    # Compute loss and gradients.
    # `nnx.value_and_grad` handles NNX modules correctly.
    # It returns grads for variables of type nnx.Param by default.
    grad_fn = nnx.value_and_grad(
        loss_fn_for_grad, argnums=0
    )  # Grad w.r.t. model (arg 0)
    loss_value, grads = grad_fn(model, batch_frame1, batch_frame2, patch_size_loss)

    # Update model parameters using the optimizer
    optimizer.update(grads)  # nnx.Optimizer updates the model's params in-place

    return (
        model,
        optimizer,
        loss_value,
        grads,
    )  # Model and optimizer are returned because JIT works with pure functions


def train_model():
    """Main training loop."""
    # Ensure reproducibility / manage randomness
    main_key = jax.random.PRNGKey(42)
    init_key, data_key = jax.random.split(main_key)

    # Initialize model and optimizer
    print("Initializing model and optimizer...")
    model, optimizer = initialize_model_and_optimizer(init_key, LEARNING_RATE)
    print("Initialization complete.")

    # Prepare TensorBoard SummaryWriter
    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)
    summary_writer = tensorboard.SummaryWriter(TENSORBOARD_LOG_DIR)

    # Prepare dataset
    print("Preparing dataset...")
    frame_pairs = get_consecutive_frame_pairs(DATASET_DIR, IMAGE_EXTENSION)
    if not frame_pairs:
        print("No frame pairs found. Please check your dataset configuration.")
        return

    train_ds = create_dataset(
        frame_pairs,
        target_height=TARGET_IMG_HEIGHT,
        target_width=TARGET_IMG_WIDTH,
        batch_size=BATCH_SIZE,
        shuffle_buffer_size=1000,  # Or a more appropriate size
        is_training=True,
    )
    # Get a JAX-compatible iterator
    train_iter = train_ds.as_numpy_iterator()
    print("Dataset prepared.")

    # Training loop
    print("Starting training...")
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        for step_in_epoch, (batch_f1_np, batch_f2_np) in enumerate(train_iter):
            # Convert NumPy arrays from tf.data to JAX arrays
            batch_f1 = jnp.asarray(batch_f1_np)
            batch_f2 = jnp.asarray(batch_f2_np)

            # Perform a training step
            # Note: JIT expects static shapes. tf.data with drop_remainder=True helps.
            # Our dummy dataset always yields full batches.
            model, optimizer, loss, grads = train_step(
                model, optimizer, batch_f1, batch_f2, PATCH_SIZE_LOSS
            )

            if global_step % LOG_EVERY_N_STEPS == 0:
                print(
                    f"Epoch: {epoch + 1}/{NUM_EPOCHS}, Step: {global_step}, Loss: {loss:.4f}"
                )
                summary_writer.scalar("train_loss", loss, global_step)
                pyramid_kernel_grads = (
                    grads.pyramid.shared_conv.kernel
                )  # Accessing the gradient for the kernel
                summary_writer.scalar(
                    "grads/pyramid_kernel_norm",
                    jnp.linalg.norm(pyramid_kernel_grads.value),
                    global_step,
                )
                summary_writer.histogram(
                    "grads/pyramid_kernel_values",
                    pyramid_kernel_grads.value,
                    global_step,
                )
                predictor_grads1 = grads.predictor.dense2.kernel
                summary_writer.scalar(
                    "grads/predictor_dense1_kernel_norm",
                    jnp.linalg.norm(predictor_grads1.value),
                    global_step,
                )
                summary_writer.histogram(
                    "grads/predictor_dense1_kernel_values",
                    predictor_grads1.value,
                    global_step,
                )

                # Log pyramid kernel weights (example for the first conv layer if accessible)
                # This requires knowing the structure of your `BaselinePyramid` params.
                # Assuming `model.pyramid.shared_conv` is the nnx.Conv module
                if hasattr(model.pyramid, "shared_conv") and isinstance(
                    model.pyramid.shared_conv, nnx.Conv
                ):
                    kernel_weights = (
                        model.pyramid.shared_conv.kernel.value
                    )  # Access .value for nnx.Param
                    summary_writer.histogram(
                        "pyramid_shared_conv_kernel", kernel_weights, global_step
                    )
                    if model.pyramid.shared_conv.bias is not None:
                        bias_weights = model.pyramid.shared_conv.bias.value
                        summary_writer.histogram(
                            "pyramid_shared_conv_bias", bias_weights, global_step
                        )
                # You can add similar logging for predictor weights

            global_step += 1

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s")

        # Reset iterator for next epoch if it's not infinite
        # tf.data datasets are typically designed to be re-iterated or re-initialized for epochs
        # If train_ds is finite, train_iter will raise StopIteration.
        # For simplicity here, re-creating iterator. Better way for large datasets is .repeat() on tf.data.Dataset
        train_iter = train_ds.as_numpy_iterator()

    summary_writer.close()
    print("Training finished.")

    # Save the final model state
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # To save, we split the model into GraphDef and State (containing params)
    # This is the NNX way to get serializable state.
    graphdef, params_state = nnx.split(model, nnx.Param)  # Filter for nnx.Param
    # Potentially save other state collections if your model uses them (e.g., nnx.BatchStat)

    # For simplicity, using JAX's save/load with npz. Orbax is better for checkpoints.
    model_save_path = os.path.join(MODEL_SAVE_DIR, "optical_flow_model.npz")
    # State is a PyTree, can be saved with np.savez
    # nnx.State is dict-like. We need to extract values to save with np.savez
    # A more robust way is to use flax.training.checkpoints or orbax

    # Simple save of params:
    # Convert params_state to a plain dict of numpy arrays for np.savez
    plain_params_dict = jax.tree_util.tree_map(
        lambda x: x if not hasattr(x, "value") else x.value, params_state
    )

    with open(model_save_path, "wb") as f:
        jnp.savez(f, **plain_params_dict)  # graphdef is not saved this way
    print(f"Model parameters saved to {model_save_path}")
    # To load:
    # 1. Recreate GraphDef: `model_for_load, _ = nnx.split(OpticalFlowNNX(...), nnx.Param)`
    #    The ... args for OpticalFlowNNX must match original construction.
    # 2. Load params: `loaded_plain_dict = jnp.load(model_save_path)`
    # 3. Create nnx.State: `loaded_params_state = nnx.State(loaded_plain_dict)`
    # 4. Merge: `loaded_model = nnx.merge(graphdef_from_step1, loaded_params_state)`
    # Or more simply, if you only save params:
    # `model_instance.update(nnx.State(loaded_plain_dict))` if model_instance is already created.


if __name__ == "__main__":
    # Ensure TF uses CPU only if no GPU is intended for data pipeline, to avoid OOM on GPU
    tf.config.set_visible_devices([], "GPU")

    train_model()
