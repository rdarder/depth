import jax
import jax.numpy as jnp

from predictor import MinimalPredictor
from upscale import upscale_and_expand_flattened_coords


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
    assert f1_coords.shape == f2_coords.shape
    assert f1_coords.shape == priors.shape
    B, H, W, C = f1.shape
    f1_selected = gather_frame_values(f1, f1_coords)
    f2_selected = gather_frame_values(f2, f2_coords)
    assert f1_selected.shape[:2] == priors.shape[:2]
    flow_with_confidence = predictor(f1_selected, f2_selected, priors)
    batched_flow_with_confidence = flow_with_confidence.reshape(B, -1, 3)
    return batched_flow_with_confidence


def get_full_coordinates_grid(shape):
    B, H, W, C = shape

    row_indices = jnp.arange(H, dtype=jnp.int32)  # Shape (H,)
    col_indices = jnp.arange(W, dtype=jnp.int32)  # Shape (W,)
    row_grid, col_grid = jnp.meshgrid(row_indices, col_indices, indexing="ij")
    row_flat = row_grid.reshape(-1)  # Shape (H*W,)
    col_flat = col_grid.reshape(-1)  # Shape (H*W,)
    grid_coords = jnp.stack([row_flat, col_flat], axis=-1)  # Shape (H*W, 2)
    batch_coords = jnp.tile(grid_coords[None, :, :], (B, 1, 1))

    return batch_coords


def estimate_flow_incrementally(
    predictor_model: MinimalPredictor,
    frame1: jax.Array,
    frame2: jax.Array,
    f1_pyramid: list[jax.Array],
    f2_pyramid: list[jax.Array],
    priors: jax.Array,
):
    assert len(f1_pyramid) == len(f2_pyramid)
    assert all(f1.shape == f2.shape for f1, f2 in zip(f1_pyramid, f2_pyramid))
    B, H, W, C = coarse_shape = f1_pyramid[0].shape
    assert frame1.shape == frame2.shape
    assert priors.shape == (B, H * W, 2)

    f1_coords = f2_coords = get_full_coordinates_grid(
        coarse_shape
    )  # only on coarse level

    f1_coords_trace = [f1_coords]
    f2_coords_trace = [f2_coords]
    f1_kept_coords_trace = []
    f2_kept_coords_trace = []
    residual_flow_confidence_trace = []

    for level, (f1, f2) in enumerate(zip(f1_pyramid, f2_pyramid)):
        residual_flow_with_confidence = estimate_flow_at_level(
            predictor_model, f1, f2, f1_coords, f2_coords, priors
        )
        coarse_img_size = jnp.array([H, W], dtype=jnp.int32)
        H = H * 2
        W = W * 2
        upscaled_focus_box = jnp.array([[0, 0], [H, W]], dtype=jnp.int32)

        f1_coords, f2_coords, priors, kept_f1_coords, kept_f2_coords = (
            upscale_and_expand_flattened_coords(
                f1_coords,
                f2_coords,
                residual_flow_with_confidence,
                upscaled_focus_box,
                coarse_img_size,
                None,
            )
        )
        f1_kept_coords_trace.append(kept_f1_coords)
        f2_kept_coords_trace.append(kept_f2_coords)
        f1_coords_trace.append(f1_coords)
        f2_coords_trace.append(f2_coords)
        residual_flow_confidence_trace.append(residual_flow_with_confidence)

    return dict(
        f1_coords=f1_coords_trace,
        f2_coords=f2_coords_trace,
        f1_kept=f1_kept_coords_trace,
        f2_kept=f2_kept_coords_trace,
        flow_with_confidence=residual_flow_confidence_trace,
    )
