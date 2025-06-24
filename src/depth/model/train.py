from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx

from datetime import datetime
from tensorboardX import SummaryWriter
import os
import orbax.checkpoint

from depth.datasets.frames import FrameSource, FramePairsDataset
from depth.inference.reflow import log_flow_grid
from depth.model.build import make_model, ModelSettings, generate_zero_priors
from depth.model.frame_pair_flow import FramePairFlow
from depth.model.loss import frame_pair_loss


@dataclass
class TrainSettings:
    learning_rate: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 100
    max_frames_lookahead: int = 10
    tensorboard_logs: Path = Path('./runs')
    checkpoint_dir: Path = Path('./checkpoints')
    train_dataset_root: Path = Path('./datasets/frames')


frame_pair_loss_with_grad = nnx.value_and_grad(frame_pair_loss, has_aux=True)


@dataclass
class Settings:
    model: ModelSettings
    train: TrainSettings


@nnx.jit
def train_step(model: FramePairFlow, optimizer: nnx.Optimizer,
               f1: jax.Array, f2: jax.Array, priors: jax.Array):
    (loss, aux), grads = frame_pair_loss_with_grad(model, f1, f2, priors)
    optimizer.update(grads)
    return loss, aux


def train_loop(settings: Settings):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(settings.train.tensorboard_logs, current_time)
    writer = SummaryWriter(log_path)

    os.makedirs(settings.train.checkpoint_dir, exist_ok=True)
    checkpoint_dir = settings.train.checkpoint_dir / current_time
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()

    print("Initializing model and optimizer...")
    model = make_model(0, train=True, settings=settings.model)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=settings.train.learning_rate)
    ))

    print("Loading dataset")
    source = FrameSource(settings.train.train_dataset_root.glob('*'),
                         img_size=settings.model.img_size,
                         cache_size=10_000)
    train_dataset = FramePairsDataset(
        source,
        batch_size=settings.train.batch_size,
        max_frame_lookahead=settings.train.max_frames_lookahead,
        include_reversed_pairs=True,
        drop_uneven_batches=True,
        seed=0
    )
    priors = generate_zero_priors(settings.train.batch_size, settings.model)

    print("Starting training loop...")
    print(f"{len(train_dataset)} frame pairs on {settings.train.batch_size} size batches "
          f"over {settings.train.num_epochs} epochs")

    global_step = 0
    for epoch in range(settings.train.num_epochs):
        print(f"Epoch {epoch}")
        for step, (f1_jax, f2_jax) in enumerate(train_dataset):
            loss_value, aux = train_step(model, optimizer, f1_jax, f2_jax, priors)
            if not jnp.isfinite(loss_value):
                print(f"Warning: NaN or Inf loss detected at step {step}. Exiting training.")
                break
            if global_step % 100 == 0:
                writer.add_scalar("train_loss", loss_value, global_step)
                print(f"Step {global_step}, Total Weighted Loss: {loss_value:.4f}")
                level_losses = [jnp.mean(flow_with_loss[:, :, :, 2])
                                for flow_with_loss in aux['flow_with_loss']]
                for level_idx, mean_unweighted_loss in enumerate(level_losses):
                    # Use clear, consistent logging tags
                    writer.add_scalar(f"train_loss_levels/{level_idx}",
                                      np.array(mean_unweighted_loss), global_step)

                log_flow_grid(aux['pyramid1'], aux['pyramid2'], aux['flow_with_loss'], writer,
                              global_step)

            global_step += 1

    print("Training finished.")
    writer.close()

    model_state = nnx.state(model)
    orbax_checkpointer.save(checkpoint_dir.absolute() / 'final', model_state)
    orbax_checkpointer.wait_until_finished()
    print("Checkpoint saving complete.")

def run():
    import tyro
    tyro.cli(train_loop)


if __name__ == '__main__':
    run()
