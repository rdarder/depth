import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import flax.nnx
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from orbax.checkpoint import StandardCheckpointer
from tensorboardX import SummaryWriter

from depth.model.build import make_model
from depth.train.build import generate_zero_priors
from depth.model.loss import frame_pair_loss
from depth.model.multi_level_flow import PyramidFlowEstimator
from depth.model.settings import Settings
from depth.training.build import make_frame_pyramids_dataset
from depth.training.log import log_train_progress

frame_pair_loss_with_grad = nnx.value_and_grad(frame_pair_loss, has_aux=True)


@nnx.jit
def train_step(model: PyramidFlowEstimator, optimizer: nnx.Optimizer,
               p1: Sequence[jax.Array], p2: Sequence[jax.Array], priors: jax.Array):
    (loss, aux), grads = frame_pair_loss_with_grad(model, p1, p2, priors)
    optimizer.update(grads)
    return loss, aux


def single_level_train_loop(model: PyramidFlowEstimator,
                            optimizer: flax.nnx.Optimizer,
                            settings: Settings,
                            logger: SummaryWriter,
                            checkpointer: StandardCheckpointer,
                            stage: int,
                            global_step: int,
                            ) -> nnx.Module:
    keep_levels = stage + 2
    train_dataset = make_frame_pyramids_dataset(settings, levels=keep_levels)
    priors = generate_zero_priors(settings.train.batch_size, settings.model)
    print(f"Starting stage {stage} training loop, using {keep_levels} pyramid levels.")
    print(f"{len(train_dataset)} frame pairs on {settings.train.batch_size} size batches "
          f"over {settings.train.num_epochs} epochs")

    for epoch in range(settings.train.num_epochs):
        print(f"Epoch {epoch}")
        for step, (f1_jax, f2_jax) in enumerate(train_dataset):
            loss_value, aux = train_step(model, optimizer, f1_jax, f2_jax, priors)
            if not jnp.isfinite(loss_value):
                print(f"Warning: NaN or Inf loss detected at step {step}. Exiting training.")
                break
            if global_step % 100 == 0:
                log_train_progress(aux, global_step, loss_value, logger)
            global_step += 1

    return global_step


def train_loop(settings: Settings):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(settings.train.tensorboard_logs, current_time)
    logger = SummaryWriter(log_path)

    os.makedirs(settings.train.checkpoint_dir, exist_ok=True)
    checkpoint_dir = settings.train.checkpoint_dir.absolute() / current_time
    checkpointer = StandardCheckpointer()
    print("Starting main training loop...")
    print("Initializing model and optimizer...")
    model = make_model(0, train=True, settings=settings.model)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=settings.train.learning_rate)
    ))
    global_step = 0
    for stage in range(settings.model.levels - 1):
        global_step = single_level_train_loop(
            model, optimizer, settings, logger, checkpointer, stage, global_step
        )
        save_checkpoint(model, checkpointer, checkpoint_dir, f"{stage}-final")
    print("Training finished.")
    logger.close()
    save_checkpoint(model, checkpointer, checkpoint_dir, 'final')


def save_checkpoint(model: PyramidFlowEstimator, checkpointer: StandardCheckpointer,
                    checkpoint_dir: Path, tag: str):
    model_state = nnx.state(model)
    checkpointer.save(checkpoint_dir / tag, model_state)
    checkpointer.wait_until_finished()
    print(f"Saved checkpoint: {tag}.")


def run():
    import tyro
    settings = tyro.cli(Settings)
    train_loop(settings)


if __name__ == '__main__':
    run()
