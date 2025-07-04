import os
from datetime import datetime
from typing import Sequence, Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from orbax.checkpoint import StandardCheckpointer
from tensorboardX import SummaryWriter

from depth.loss.frame_pair_pyramid_loss import frame_pair_pyramid_loss_value_and_grad, \
    LossTrace
from depth.model.build import make_model
from depth.train.build import generate_zero_priors
from depth.model.pyramid_flow import PyramidFlowEstimator
from depth.model.settings import Settings
from depth.training.build import make_frame_pyramids_dataset
from depth.training.log import log_train_progress


@nnx.jit
def train_step(model: PyramidFlowEstimator,
               optimizer: nnx.Optimizer,
               p1: Sequence[jax.Array],
               p2: Sequence[jax.Array],
               priors: jax.Array) -> LossTrace:
    (loss, loss_trace), grads = frame_pair_pyramid_loss_value_and_grad(model, p1, p2, priors)
    optimizer.update(grads)
    return loss_trace


class Train:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._checkpointer = StandardCheckpointer()
        log_path = os.path.join(self._settings.train.tensorboard_logs, self._run_id)
        self._logger = SummaryWriter(log_path)
        print("Initializing model and optimizer...")
        self._model = make_model(0, train=True, settings=self._settings.model)
        transformations = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=self._settings.train.learning_rate),
            optax.masked(optax.keep_params_nonnegative(), self._build_spatial_conv_mask()),
        )
        self._optimizer = nnx.Optimizer(self._model, transformations)

    def run(self):
        print("Starting main training loop...")
        global_step = 0
        epochs = self._settings.train.num_epochs
        for stage in range(self._settings.model.levels - 1):
            global_step = self.single_level_train_loop(stage, global_step, epochs)
            epochs = epochs + 2
            self._save_checkpoint(f"{stage}-final")
        print("Training finished.")
        self._logger.close()
        self._save_checkpoint(f"final")

    def single_level_train_loop(self, stage: int, global_step: int, epochs: int) -> int:
        keep_levels = stage + 2
        train_dataset = make_frame_pyramids_dataset(self._settings, levels=keep_levels)
        priors = generate_zero_priors(self._settings.train.batch_size, self._settings.model)
        print(f"Starting stage {stage} training loop, using {keep_levels} pyramid levels.")
        print(f"{len(train_dataset)} frame pairs on {self._settings.train.batch_size} size batches "
              f"over {self._settings.train.num_epochs} epochs")

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for step, (f1_jax, f2_jax) in enumerate(train_dataset):
                trace = train_step(self._model, self._optimizer, f1_jax, f2_jax, priors)
                if not jnp.isfinite(trace.weighted_loss):
                    print(f"Warning: NaN or Inf loss detected at step {step}. Exiting training.")
                    return
                if global_step % 100 == 0:
                    log_train_progress(trace, global_step, self._logger)
                global_step += 1

        return global_step

    def _save_checkpoint(self, tag: str):
        os.makedirs(self._settings.train.checkpoint_dir, exist_ok=True)
        checkpoint_dir = self._settings.train.checkpoint_dir.absolute() / self._run_id
        model_state = nnx.state(self._model)
        self._checkpointer.save(checkpoint_dir / tag, model_state)
        self._checkpointer.wait_until_finished()
        print(f"Saved checkpoint: {self._run_id}/{tag}.")

    def _build_spatial_conv_mask(self):
        _, state, _ = nnx.split(self._model, nnx.Param, ...)

        def mask_fn(path, s) -> bool:
            return path == ('_upsampler', 'spatial_influence_conv', 'kernel')

        mapped_state = nnx.map_state(mask_fn, state)
        # ensure we matched at least one time. else we probably renamed the model's convolution.
        assert any([s[1] for s in nnx.to_flat_state(mapped_state)])

        return mapped_state


def run():
    import tyro
    settings = tyro.cli(Settings)
    Train(settings).run()


if __name__ == '__main__':
    run()
