from typing import Optional

import jax
import jax.lax
import jax.numpy as jnp


def upscale_and_expand_flattened_coords(
    coarse_f1_coordinates: jax.Array,  # Shape: [B, N, 2] (int)
    coarse_flow: jax.Array,  # Shape: [B, N, 2] (float)
    coarse_f2_coordinates: jax.Array,  # Shape: [B, N, 2] (int)
    confidence: jax.Array,  # Shape: [B, N] (float)
    fine_focus_box: jax.Array,  # Shape [2,2] (int)
    coarse_size: jax.Array,  # Shape [2] (int)
    select_top: Optional[int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    B, N, D = coarse_f2_coordinates.shape

    assert coarse_f1_coordinates.shape == coarse_f2_coordinates.shape
    assert D == 2
    if select_top is None:
        select_top = N
    assert select_top <= N
    assert coarse_f1_coordinates.dtype == jnp.int32
    assert coarse_f2_coordinates.dtype == jnp.int32
    assert coarse_size.dtype == jnp.int32
    assert coarse_size.shape == (2,)
    assert fine_focus_box.dtype == jnp.int32
    assert fine_focus_box.shape == (2, 2)

    flowed_f2 = coarse_f2_coordinates + coarse_flow  # [B, N, 2] # float
    upscaled_flowed_f2 = flowed_f2 * 2  # [B, N, 2] # float
    upscaled_f1 = coarse_f1_coordinates * 2  # [B, N, 2] # int
    rounded_upscaled_flowed_f2 = jnp.round(upscaled_flowed_f2).astype(
        jnp.int32
    )  # [B, N, 2] #int
    upscaled_f2_priors = (
        upscaled_flowed_f2 - rounded_upscaled_flowed_f2
    )  # [B, N, 2] # float

    if select_top < N:
        box_scores = get_box_score(upscaled_flowed_f2, fine_focus_box)
        # print(box_scores)

        exclusion_score = box_scores * confidence

        kept_coords_indices = get_top_k_indices(exclusion_score, select_top)
        kept_coords_indices = jnp.expand_dims(kept_coords_indices, axis=-1)

        kept_rounded_upscaled_flowed_f2 = jnp.take_along_axis(
            rounded_upscaled_flowed_f2, kept_coords_indices, axis=1
        )
        kept_upscaled_f1 = jnp.take_along_axis(upscaled_f1, kept_coords_indices, axis=1)
        # print(kept_rounded_upscaled_flowed_f2_coords)
        # kept_rounded_upscaled_flowed_f2 = jnp.take_along_axis(
        #     upscaled_f2_priors, kept_rounded_upscaled_flowed_f2_coords, axis=1
        # )
        kept_upscaled_f2_priors = jnp.take_along_axis(
            upscaled_f2_priors, kept_coords_indices, axis=1
        )

        kept_coarse_f1 = jnp.take_along_axis(
            coarse_f1_coordinates, kept_coords_indices, axis=1
        )

        kept_coarse_f2 = jnp.take_along_axis(
            coarse_f2_coordinates, kept_coords_indices, axis=1
        )

    else:
        kept_upscaled_f1 = upscaled_f1
        kept_rounded_upscaled_flowed_f2 = rounded_upscaled_flowed_f2
        kept_upscaled_f2_priors = upscaled_f2_priors
        kept_coarse_f1 = coarse_f1_coordinates
        kept_coarse_f2 = coarse_f2_coordinates

    # 3. Expand the prior to the 4*N output points
    # [B, N, 2] -> [B, N, 1, 2] -> [B, N, 4, 2] (repeat) -> [B, N*4, 2] (reshape)
    expanded_upscaled_f2_priors = jnp.repeat(
        kept_upscaled_f2_priors, repeats=4, axis=1
    )  # [B, 4*N, 2]

    # 4. Generate coordinates for the 2x2 fine grid cells for each of the N input points
    expanded_kept_rounded_upscaled_flowed_f2 = jnp.repeat(
        kept_rounded_upscaled_flowed_f2, repeats=4, axis=1
    )  # [B, 4*N, 2]

    expanded_kept_upscaled_f1 = jnp.repeat(kept_upscaled_f1, repeats=4, axis=1)

    # Create the 2x2 offsets: (0,0), (1,0), (0,1), (1,1) for (dx, dy) or (dc, dr)
    pixel_offsets = jnp.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=jnp.int32
    )  # Shape [4, 2]

    # Tile these offsets for each of the N input points.
    # We want the pattern [off0,off1,off2,off3, off0,off1,off2,off3, ...]
    # So, tile along a new axis for N, then reshape.
    # [4,2] -> [1, 4, 2] -> [N, 4, 2] (tile) -> [N*4, 2] (reshape)
    tiled_pixel_offsets = jnp.tile(pixel_offsets, (select_top, 1))  # Shape [4*N, 2]
    # Add batch dimension for broadcasting: [1, 4*N, 2]
    batched_tiled_pixel_offsets = jnp.expand_dims(tiled_pixel_offsets, axis=0)
    # This will broadcast over B if B > 1.

    # Add offsets to the expanded base coordinates
    offset_expanded_kept_rounded_upscaled_flowed_f2 = (
        expanded_kept_rounded_upscaled_flowed_f2 + batched_tiled_pixel_offsets
    )
    offset_expanded_kept_upscaled_f1 = (
        expanded_kept_upscaled_f1 + batched_tiled_pixel_offsets
    )
    # Shape: [B, 4*N, 2]
    #

    clipped_offset_expanded_kept_rounded_upscaled_flowed_f2 = jnp.clip(
        offset_expanded_kept_rounded_upscaled_flowed_f2,
        jnp.zeros(2, dtype=jnp.int32),
        2 * coarse_size - 1,
    )

    # Shape: [B, 4*N, 2]
    return (
        offset_expanded_kept_upscaled_f1,
        clipped_offset_expanded_kept_rounded_upscaled_flowed_f2,
        expanded_upscaled_f2_priors,
        kept_coarse_f1,
        kept_coarse_f2,
    )


def get_box_score(coordinates: jax.Array, box: jax.Array):
    """Returns whether or not each coordinate fits inside a given box.
    The coordinates can be int or float.
    The box is expected to be int. Regardless, it does an inclusive min and exclusive max.
    """
    # coordinates: [B, N, 2]
    # box: [2, 2]
    # box: [y_min, x_min], [y_max, x_max]

    min_limits = box[0, :]
    max_limits = box[1, :]
    gte_min_per_channel = jnp.greater_equal(coordinates, min_limits)
    lt_max_per_channel = jnp.less(coordinates, max_limits)
    separate_coordinate_inside_box = jnp.logical_and(
        gte_min_per_channel, lt_max_per_channel
    )
    return jnp.all(separate_coordinate_inside_box, axis=-1)


# @partial(jax.jit, static_argnames=("k", "recall_target"))
def get_top_k_indices(array: jax.Array, k: int, recall_target=0.95):
    _, indices = jax.lax.approx_max_k(array, k, recall_target=recall_target)
    return indices
