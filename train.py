import os
import time
from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf  # For tf.data pipeline
from jax.tree_util import tree_map_with_path

from datasets import create_dataset, get_consecutive_frame_pairs
from flow_model import OpticalFlow
from loss import model_loss
from datetime import datetime

# --- Training Configuration ---
LEARNING_RATE = 1e-9
NUM_EPOCHS = 500
BATCH_SIZE = 1000  # Should match dataset batch size
NUM_PYRAMID_LEVELS = 4
PREDICTOR_HIDDEN_FEATURES = (16,8,8)
PATCH_SIZE_LOSS = 3
PATCH_SIZE_PYRAMID = 4
PYRAMID_CHANNELS = 6
LOG_EVERY_N_STEPS = 10
TENSORBOARD_LOG_DIR_BASE = "./tensorboard_logs"
MODEL_SAVE_DIR = "./saved_models"

# Dataset params (should match dataset script)
DATASET_DIR = "datasets/frames"  # or your actual dataset path
IMAGE_EXTENSION = ".png"
TARGET_IMG_HEIGHT = 64  # Example, make sure divisible for pyramid
TARGET_IMG_WIDTH = 64  # Example
FLOW_REGULARIZATION_WEIGHT = 2.0  # You might need to tune this value


def initialize_model_and_optimizer(key: jax.random.PRNGKey, learning_rate: float):
    """Initializes the model and the optimizer."""
    model_key, _ = jax.random.split(key)

    # Initialize the main model (which initializes submodules)
    # NNX modules are stateful, initialization happens at construction.
    # rngs=nnx.Rngs(params=model_key) helps if submodules need different keys based on collection.
    # For simple nnx.Param, just one key is often fine.
    model = OpticalFlow(
        num_pyramid_levels=NUM_PYRAMID_LEVELS,
        pyramid_patch_size=PATCH_SIZE_PYRAMID,
        predictor_hidden_features=PREDICTOR_HIDDEN_FEATURES,
        pyramid_output_channels=PYRAMID_CHANNELS,
        rngs=nnx.Rngs(params=model_key),  # Pass Rngs for parameter initialization
    )

    # Create the optimizer using nnx.Optimizer
    # It will manage the parameters of the 'model' instance.
    tx = optax.adamw(learning_rate=learning_rate)
    optimizer = nnx.Optimizer(model, tx)

    return model, optimizer


@partial(nnx.jit, static_argnums=(4, 5))  # JIT compile the training step
def train_step(
    model: OpticalFlow,
    optimizer: nnx.Optimizer,
    batch_frame1: jax.Array,
    batch_frame2: jax.Array,
    loss_patch_size: int,
    flow_regularization_weight: float,
):
    """Performs a single training step."""
    # Compute loss and gradients.
    # `nnx.value_and_grad` handles NNX modules correctly.
    # It returns grads for variables of type nnx.Param by default.
    grad_fn = nnx.value_and_grad(model_loss, argnums=0, has_aux=True)  # Grad w.r.t. model (arg 0)
    (loss_value, debug_info), grads = grad_fn(
        model,
        batch_frame1,
        batch_frame2,
        loss_patch_size,
        flow_regularization_weight=flow_regularization_weight,
    )

    # Update model parameters using the optimizer
    optimizer.update(grads)  # nnx.Optimizer updates the model's params in-place

    return (
        model,
        optimizer,
        loss_value,
        grads,
        debug_info
    )  # Model and optimizer are returned because JIT works with pure functions


def train_model():
    """Main training loop."""
    # Ensure reproducibility / manage randomness
    main_key = jax.random.PRNGKey(1109)
    init_key, data_key = jax.random.split(main_key)

    # Initialize model and optimizer
    print("Initializing model and optimizer...")
    model, optimizer = initialize_model_and_optimizer(init_key, LEARNING_RATE)
    print("Initialization complete.")

    # Prepare TensorBoard SummaryWriter
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_run_log_dir = os.path.join(TENSORBOARD_LOG_DIR_BASE, current_time)
    if not os.path.exists(current_run_log_dir):
        os.makedirs(current_run_log_dir)
    summary_writer = tf.summary.create_file_writer(current_run_log_dir)


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
            model, optimizer, loss, grads, debug_info = train_step(
                model,
                optimizer,
                batch_f1,
                batch_f2,
                PATCH_SIZE_LOSS,
                FLOW_REGULARIZATION_WEIGHT,
            )

            if global_step % LOG_EVERY_N_STEPS == 0:
                level_loss_weights = [
                    p["loss_weight"].item() for p in debug_info["photometric"][
                    "per_level"]
                ]
                formatted_weights = [f"{w:4f}" for w in level_loss_weights]

                print(
                    f"Epoch: {epoch + 1}/{NUM_EPOCHS}, Step: {global_step}, Loss: {loss:.10f}"
                    f", Level weights: {formatted_weights}"
                )
                with summary_writer.as_default():
                    tf.summary.scalar("train_loss", loss, global_step)
                    # log_gradients_to_tensorboard(grads, model, global_step)
                    log_debug_info_to_tensorboard(debug_info, global_step, summary_writer)
                summary_writer.flush()

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


def log_gradients_to_tensorboard(grads, model, step):
    """Logs gradient statistics and histograms to TensorBoard."""

    print(f"Logging gradients at step {step}...")  # Optional debug print

    # Use tree_map_with_path to traverse the gradients Pytree
    def log_grad_leaf(path, grad_value: jnp.ndarray):
        # Convert JAX array to TensorFlow tensor
        tf_grad = tf.convert_to_tensor(grad_value)

        # Create a unique name from the path (e.g., 'model/pyramid/shared_conv/kernel')
        # Remove the leading 'params' if your grads are from state.params()
        # Or keep it if it makes sense for your Pytree structure
        path_parts = tuple(str(p) for p in path if str(p) != "params")  # Example filter
        name = "/".join(path_parts)  # Use '/' as separator for TB hierarchy

        if name:  # Avoid logging empty names or top-level containers if any
            try:
                # Log gradient histogram
                tf.summary.histogram(f"gradients/{name}", tf_grad, step=step)

                # Log gradient norms as scalars
                norm = tf.norm(tf_grad)
                tf.summary.scalar(f"gradients_norms/{name}/norm", norm, step=step)

                # Log mean/max absolute values as scalars
                # mean_abs = tf.reduce_mean(tf.abs(tf_grad))
                # max_abs = tf.reduce_max(tf.abs(tf_grad))
                # tf.summary.scalar(f"gradients/{name}/mean_abs", mean_abs, step=step)
                # tf.summary.scalar(f"gradients/{name}/max_abs", max_abs, step=step)

            except Exception as e:
                # Handle potential errors during logging (e.g., empty tensors)
                print(f"Error logging gradient for path {name}: {e}")

        return None  # tree_map_with_path needs to return something, but we discard the result

    # Apply the logging function to each leaf in the grads Pytree
    tree_map_with_path(log_grad_leaf, grads)

    # Optional: Log overall gradient norm
    global_norm = optax.global_norm(grads)  # Optax provides a utility for this
    tf.summary.scalar("gradients/global_norm", global_norm, step=step)

    pyramid_kernel_grads = (
        grads.pyramid.shared_conv.kernel
    )  # Accessing the gradient for the kernel
    tf.summary.scalar(
        "grads/pyramid_kernel",
        jnp.linalg.norm(pyramid_kernel_grads.value),
        step,
    )
    for i, layer in enumerate(grads.predictor.hidden_layers):
        tf.summary.scalar(
            f"grads/predictor_dense_{i:02}",
            jnp.linalg.norm(layer.kernel.value)
        )

    pyramid_weights = (
        model.pyramid.shared_conv.kernel.value
    )  # Access .value for nnx.Param
    tf.summary.histogram("weights/pyramid_kernel", pyramid_weights, step)
    bias_weights = model.pyramid.shared_conv.bias.value
    tf.summary.histogram("weights/pyramid_bias", bias_weights, step)

    for i, layer in enumerate(grads.predictor.hidden_layers):
        tf.summary.histogram(
            f"weights/predictor_dense_{i:02}",
            layer.kernel.value
        )

    # Flush the writer to disk


def log_debug_info_to_tensorboard(aux_data, step, summary_writer):
    """Logs auxiliary data from the loss function to TensorBoard."""
    with summary_writer.as_default():
        # Log components of the main loss
        tf.summary.scalar(
            "loss_components/photometric_loss",
            aux_data["photometric_loss_component"],
            step=step,
        )
        tf.summary.scalar(
            "loss_components/flow_regularization_loss",
            aux_data["flow_regularization_loss_component"],
            step=step,
        )
        tf.summary.scalar(
            "loss_components/avg_flow_squared", aux_data["avg_flow_squared"], step=step
        )

        # Log details from photometric_loss
        photo_details = aux_data.get("photometric", {})
        tf.summary.scalar(
            "photometric_debug/loss",
            photo_details.get("loss", 0.0),
            step=step,
        )
        tf.summary.scalar(
            "photometric_debug/total_in_frame",
            photo_details.get("total_in_frame", 0.0),
            step=step,
        )

        per_level_details_list = photo_details.get("per_level", [])
        # Note: per_level_details_list is a Python list of dicts of JAX arrays.
        # When JITted, these JAX arrays are concrete values *outside* the JIT boundary.
        for level_detail in per_level_details_list:
            level_idx = level_detail.get(
                "level_idx"
            )  # This will be a concrete Python int
            # Convert JAX arrays to NumPy for TF summary if they are still JAX arrays here
            # (depends on how JAX handles dicts with JAX arrays as values through has_aux)
            # Typically, they are returned as JAX arrays.
            unweighted_avg_loss = jnp.asarray(
                level_detail.get("unweighted_avg_loss", 0.0)
            )
            loss_weight = jnp.asarray(
                level_detail.get("loss_weight", 0.0)
            )

            tf.summary.scalar(
                f"photometric_level_{level_idx}/unweighted_avg_loss",
                unweighted_avg_loss,
                step=step,
            )
            tf.summary.scalar(
                f"photometric_level_{level_idx}/loss_weight",
                loss_weight,
                step=step,
            )
            tf.summary.scalar(
                f"photometric_level_{level_idx}/in_frame",
                jnp.asarray(level_detail.get("in_frame", 0)),
                step=step,
            )
            tf.summary.scalar(
                f"photometric_level_{level_idx}/loss",
                jnp.asarray(level_detail.get("loss", 0)),
                step=step,
            )


if __name__ == "__main__":
    # Ensure TF uses CPU only if no GPU is intended for data pipeline, to avoid OOM on GPU
    tf.config.set_visible_devices([], "GPU")

    train_model()


