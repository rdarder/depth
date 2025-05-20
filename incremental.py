import flax.nnx as nnx
import jax
import jax.numpy as jnp

from predictor import MinimalPredictor
from pyramid import BaselinePyramid


def hierarchical_flow_estimation(
    frame1_features_pyramid: list[
        jax.Array
    ],  # Coarsest first (idx 0) to finest (idx num_levels-1)
    frame2_features_pyramid: list[
        jax.Array
    ],  # Coarsest first (idx 0) to finest (idx num_levels-1)
    predictor_module: MinimalPredictor,  # Or actual MinimalPredictorNNX
    num_levels: int,
):
    """
    Estimates dense optical flow hierarchically from coarse to fine.

    Args:
        frame1_features_pyramid: List of [B, H_l, W_l, C_feat] features for frame 1,
                                 ordered from coarsest (index num_levels-1) to finest (0).
        frame2_features_pyramid: List of [B, H_l, W_l, C_feat] features for frame 2,
                                 ordered similarly.
        predictor_module: The learned motion predictor module.
        num_levels: The number of pyramid levels.
    Returns:
        final_dense_flow: A JAX array of shape [B, H_0, W_0, 2] representing
                          the estimated flow at the finest level resolution.
    """
    # Initial batch size and finest level dimensions
    B = frame1_features_pyramid[0].shape[0]
    H_0 = frame1_features_pyramid[0].shape[1]
    W_0 = frame1_features_pyramid[0].shape[2]

    # Initialize loop state at the coarsest level (pyramid index -1)
    F1_coarsest = frame1_features_pyramid[-1]
    _, H_coarsest, W_coarsest, _ = F1_coarsest.shape

    # Create initial grid of points for the coarsest level
    # (batch_idx, r_idx, c_idx)
    batch_coords, r_coords, c_coords = jnp.meshgrid(
        jnp.arange(B), jnp.arange(H_coarsest), jnp.arange(W_coarsest), indexing="ij"
    )

    loop_batch_indices = batch_coords.flatten()
    loop_r_coords = r_coords.flatten().astype(jnp.int32)
    loop_c_coords = c_coords.flatten().astype(jnp.int32)

    num_initial_points = B * H_coarsest * W_coarsest
    loop_accumulated_flow = jnp.zeros((num_initial_points, 2), dtype=jnp.float32)

    # Loop from coarsest (L_idx=0) to finest (L_idx=num_levels-1)
    for L_idx in range(num_levels - 1, -1, -1):
        F1_L = frame1_features_pyramid[L_idx]
        F2_L = frame2_features_pyramid[L_idx]
        _, H_L, W_L, _ = F1_L.shape  # Current level dimensions

        # 1. Calculate predictor prior (subpixel part of accumulated flow)
        integer_part_of_accum_flow = jnp.round(loop_accumulated_flow)
        predictor_prior = loop_accumulated_flow - integer_part_of_accum_flow

        # 2. Gather F1 features
        # loop_r_coords and loop_c_coords are already valid for F1_L at this stage
        # because they either come from the initial grid or the 2* mapping from previous.
        # We need to ensure they are clipped if they somehow go out of bounds due to
        # extreme flow estimates previously, though with 2* mapping they should be fine
        # for the grid itself.
        # For safety, clip, though for F1, coords should be "pristine" grid coords.
        safe_r_f1 = jnp.clip(loop_r_coords, 0, H_L - 1)
        safe_c_f1 = jnp.clip(loop_c_coords, 0, W_L - 1)
        gathered_f1 = F1_L[loop_batch_indices, safe_r_f1, safe_c_f1, :]

        # 3. Warp and Gather F2 features
        # Warped coordinates are relative to the F1 grid points
        warped_r_float = (
            loop_r_coords.astype(jnp.float32) + integer_part_of_accum_flow[:, 1]
        )  # uy
        warped_c_float = (
            loop_c_coords.astype(jnp.float32) + integer_part_of_accum_flow[:, 0]
        )  # ux

        # Round and clip warped coordinates for F2 sampling
        warped_r_int = jnp.clip(jnp.round(warped_r_float).astype(jnp.int32), 0, H_L - 1)
        warped_c_int = jnp.clip(jnp.round(warped_c_float).astype(jnp.int32), 0, W_L - 1)
        gathered_f2 = F2_L[loop_batch_indices, warped_r_int, warped_c_int, :]

        # 4. Form predictor input
        predictor_input = jnp.concatenate(
            [gathered_f1, gathered_f2, predictor_prior], axis=-1
        )

        # 5. Predict delta flow
        delta_flow = predictor_module(predictor_input)

        # 6. Calculate flow increment at this level (relative to integer warp)
        flow_increment_at_level = predictor_prior + delta_flow

        # 7. Update total accumulated flow (at current level L's scale)
        current_total_flow_L_scale = (
            integer_part_of_accum_flow + flow_increment_at_level
        )

        # 8. If not the finest level, prepare for the next (finer) level
        if L_idx > 0:
            num_points_at_L = loop_batch_indices.shape[0]

            next_loop_batch_indices = jnp.repeat(loop_batch_indices, 4, axis=0)

            base_r_next = loop_r_coords * 2
            base_c_next = loop_c_coords * 2

            # Offsets for the 2x2 block in the finer grid
            # tile: [0,1,0,1] -> [0,1,0,1,0,1,0,1,...]
            # repeat: [r0,r0,r0,r0, r1,r1,r1,r1,...]
            r_offsets = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
            c_offsets = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

            next_loop_r_coords = jnp.tile(r_offsets, num_points_at_L) + jnp.repeat(
                base_r_next, 4, axis=0
            )
            next_loop_c_coords = jnp.tile(c_offsets, num_points_at_L) + jnp.repeat(
                base_c_next, 4, axis=0
            )

            # Scale up the accumulated flow values for the finer level's coordinate system
            next_loop_accumulated_flow = (
                jnp.repeat(current_total_flow_L_scale, 4, axis=0) * 2.0
            )

            # Update loop variables for the next iteration
            loop_batch_indices = next_loop_batch_indices
            loop_r_coords = next_loop_r_coords
            loop_c_coords = next_loop_c_coords
            loop_accumulated_flow = next_loop_accumulated_flow
        else:
            # This is the finest level (L_idx == num_levels - 1)
            # current_total_flow_L_scale is the final flow for the processed locations
            # loop_batch_indices, loop_r_coords, loop_c_coords define these locations
            pass  # Final values are already computed

    # After the loop, reconstruct the dense flow field at original resolution (H_0, W_0)
    # loop_batch_indices, loop_r_coords, loop_c_coords are now for level 0 (finest)
    # current_total_flow_L_scale contains the flow values for these points

    final_dense_flow = jnp.zeros((B, H_0, W_0, 2), dtype=jnp.float32)
    final_dense_flow = final_dense_flow.at[
        loop_batch_indices, loop_r_coords, loop_c_coords
    ].set(current_total_flow_L_scale)

    return final_dense_flow


# Example Usage:
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    rngs_pyramid = nnx.Rngs(params=key)  # Dummy, not used by dummy pyramid
    rngs_predictor = nnx.Rngs(params=jax.random.PRNGKey(1))  # Dummy

    # Create dummy pyramid and predictor
    num_pyr_levels = 3
    dummy_pyramid_builder = BaselinePyramid(
        num_levels=num_pyr_levels, rngs=rngs_pyramid
    )
    dummy_predictor = MinimalPredictor(rngs=rngs_predictor)

    # Create dummy input frames
    # Finest level (Level 0) is 32x32 for this example
    # Coarsest level (Level 2) will be 32/(2^2) = 8x8
    # Pyramid features will be: L0 (idx 2): 32x32, L1 (idx 1): 16x16, L2 (idx 0): 8x8
    img_h0, img_w0 = 32, 32
    batch_size = 1
    dummy_frame1 = jnp.zeros((batch_size, img_h0, img_w0, 1), dtype=jnp.float32)
    dummy_frame2 = jnp.ones(
        (batch_size, img_h0, img_w0, 1), dtype=jnp.float32
    )  # Shifted frame

    # Generate feature pyramids (coarsest first)
    # The dummy pyramid returns them coarsest-first if I reverse it.
    # My current hierarchical loop expects coarsest first.
    # The dummy pyramid code was: return list(reversed(pyramid)) # Coarsest first
    # This means pyramid_f1[0] is coarsest, pyramid_f1[num_levels-1] is finest.

    pyramid_f1_raw = dummy_pyramid_builder(dummy_frame1)  # finest to coarsest
    pyramid_f2_raw = dummy_pyramid_builder(dummy_frame2)

    # The hierarchical_flow_estimation expects coarsest first.
    # DummyBaselinePyramidNNX already reverses it to be coarsest first if I did that right.
    # Check dummy pyramid: `pyramid.append(level_output)` (finest first), then `list(reversed(pyramid))`
    # So, pyramid_f1_raw[0] is coarsest. This is correct for the loop.

    print("Pyramid features (coarsest first):")
    for i in range(num_pyr_levels):
        print(
            f"  Level {i} (from coarse): F1 shape {pyramid_f1_raw[i].shape}, F2 shape {pyramid_f2_raw[i].shape}"
        )

    # Estimate flow
    estimated_flow = hierarchical_flow_estimation(
        pyramid_f1_raw, pyramid_f2_raw, dummy_predictor, num_pyr_levels
    )

    print(f"\nEstimated dense flow shape: {estimated_flow.shape}")
    # Since dummy predictor outputs zero delta, and initial accum flow is zero,
    # prior is zero, flow_increment is zero, total_flow_L_scale is zero.
    # So final flow should be all zeros.
    print(f"Sum of absolute flow values: {jnp.sum(jnp.abs(estimated_flow))}")

    # Test with a predictor that does something:
    class ShiftingPredictor(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs | None = None):
            pass

        def __call__(self, inputs: jax.Array) -> jax.Array:
            # inputs are [..., f1_0:4, f2_0:4, prior_0:2]
            prior = inputs[..., 8:10]
            # Let's make delta = [0.1, 0.1] - prior, so flow_increment = [0.1, 0.1]
            delta = jnp.ones_like(prior) * 0.1 - prior
            return delta

    shifting_predictor = ShiftingPredictor()
    estimated_flow_shifting = hierarchical_flow_estimation(
        pyramid_f1_raw, pyramid_f2_raw, shifting_predictor, num_pyr_levels
    )
    print("\nWith shifting predictor:")
    print(f"Estimated dense flow shape: {estimated_flow_shifting.shape}")
    # Expected flow:
    # L2 (coarsest, 8x8): prior=0, delta=0.1. total_L_scale = 0 + (0+0.1) = 0.1. accum_for_L1 = 0.1*2 = 0.2
    # L1 (16x16): accum=0.2. prior=0.2-0=0.2. delta=0.1-0.2 = -0.1. total_L_scale = 0 + (0.2-0.1) = 0.1. accum_for_L0 = 0.1*2=0.2
    # L0 (32x32): accum=0.2. prior=0.2. delta=-0.1. total_L_scale = 0 + (0.2-0.1) = 0.1.
    # So final flow should be 0.1 everywhere.
    print(f"Example flow values (first point): {estimated_flow_shifting[0, 0, 0, :]}")
    print(f"all flow estimation values: {estimated_flow_shifting}")
    print(f"Sum of absolute flow values: {jnp.sum(jnp.abs(estimated_flow_shifting))}")
    # Expected sum for 0.1: B * H0 * W0 * 2 * 0.1 = 1 * 32 * 32 * 2 * 0.1 = 204.8
    # My logic trace above implies final flow of 0.1. Let's re-trace the ShiftingPredictor.
    # L_idx = 0 (coarsest, e.g. 8x8 for 3 levels, H_0=32)
    #   loop_accumulated_flow = 0
    #   integer_part_of_accum_flow = 0
    #   predictor_prior = 0
    #   delta_flow = 0.1 - 0 = 0.1
    #   flow_increment_at_level = 0 + 0.1 = 0.1
    #   current_total_flow_L_scale = 0 + 0.1 = 0.1
    #   L_idx (0) < num_levels-1 (2), so map.
    #   next_loop_accumulated_flow = jnp.repeat(0.1, 4) * 2.0 = 0.2 (for all new points)
    # L_idx = 1 (middle, 16x16)
    #   loop_accumulated_flow = 0.2
    #   integer_part_of_accum_flow = 0 (since 0.2 rounds to 0)
    #   predictor_prior = 0.2 - 0 = 0.2
    #   delta_flow = 0.1 - 0.2 = -0.1
    #   flow_increment_at_level = 0.2 + (-0.1) = 0.1
    #   current_total_flow_L_scale = 0 + 0.1 = 0.1
    #   L_idx (1) < num_levels-1 (2), so map.
    #   next_loop_accumulated_flow = jnp.repeat(0.1, 4) * 2.0 = 0.2 (for all new points)
    # L_idx = 2 (finest, 32x32)
    #   loop_accumulated_flow = 0.2
    #   integer_part_of_accum_flow = 0
    #   predictor_prior = 0.2
    #   delta_flow = 0.1 - 0.2 = -0.1
    #   flow_increment_at_level = 0.2 + (-0.1) = 0.1
    #   current_total_flow_L_scale = 0 + 0.1 = 0.1
    #   L_idx (2) == num_levels-1 (2). Loop ends.
    # Final flow is current_total_flow_L_scale = 0.1. This looks correct.
