import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MinimalPredictor(nnx.Module):
    """
    Minimal learned per-pixel motion predictor (MLP) with flow prior.

    Takes concatenated features from two frames at the same location,
    plus a 2D flow prior, and predicts a 2D flow residual and confidence.
    """

    def __init__(
            self,
            input_features: int = 10,
            hidden_features: tuple[int, ...] = (32, 32),  # Changed from int to tuple
            output_features: int = 3,  # 2D residual flow + 1D confidence
            *,
            rngs: nnx.Rngs,
    ):
        # Input features:
        #   e.g., 4 from frame1 features
        # + 4 from frame2 features
        # + 2 from flow prior (prior_ux, prior_uy)
        # = 10 total input features (example)
        #
        # Output features: 3 (delta_ux, delta_uy, confidence)

        self.hidden_layers = []
        current_in_features = input_features

        for i, h_feats in enumerate(hidden_features):
            layer = nnx.Linear(
                in_features=current_in_features,
                out_features=h_feats,
                rngs=rngs,
            )
            self.hidden_layers.append(layer)
            current_in_features = h_feats

        self.output_layer = nnx.Linear(
            in_features=current_in_features,  # Input from the last hidden layer
            out_features=output_features,
            rngs=rngs,
        )

    def __call__(
            self, f1: jax.Array, f2: jax.Array, priors: jax.Array
    ) -> jax.Array:
        """
        Applies the MLP prediction.

        Args:
            f1: Features from frame 1. Shape [B', num_f1_features].
            f2: Features from frame 2. Shape [B', num_f2_features].
            priors: Flow priors. Shape [B', num_prior_features (e.g., 2)].

        Returns:
            Predicted flow residuals and confidence. Shape [B', output_features (e.g., 3)].
        """
        x = jnp.concatenate([f1, f2, priors], axis=-1)

        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.relu(x)

        flow_with_confidence = self.output_layer(x)
        return flow_with_confidence
