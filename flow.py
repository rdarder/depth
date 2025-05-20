import jax.numpy as jnp

from predictor import MinimalPredictor


def gather_frame_values(frame, coords):
    B, H, W, C = frame.shape
    BC, N, P = coords.shape
    assert BC == B
    assert P == 2
    batch_indices = jnp.expand_dims(jnp.arange(B), -1)
    row_indices = coords[:, :, 0]
    col_indices = coords[:, :, 1]
    gathered = frame[batch_indices, row_indices, col_indices]
    assert gathered.shape == (B, N, C)
    return gathered


def estimate_flow_at_level(
    predictor: MinimalPredictor, f1, f2, f1_coords, f2_coords, priors
):
    assert f1.shape == f2.shape
    assert f1.shape == priors.shape
    B, H, W, C = f1.shape
    f1_selected = gather_frame_values(f1, f1_coords)
    f2_selected = gather_frame_values(f2, f2_coords)
    concatenated_inputs = jnp.concatenate([f1_selected, f2_selected, priors], axis=-1)
    predictor_input = concatenated_inputs.reshape(-1, concatenated_inputs.shape[-1])
    predicted_flow = predictor(predictor_input)
    batched_predicted_flow = predicted_flow.reshape(
        B, -1, concatenated_inputs.shape[-1]
    )
    return batched_predicted_flow
