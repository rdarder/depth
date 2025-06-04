from pathlib import Path
from typing import Sequence

import flax.nnx
from PIL import Image
import pywt
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from flax import nnx


class PatchFlowPredictor(nnx.Module):
    """
    Predictor network for estimating residual optical flow within a patch.

    Inputs:
    - patch1: patch from frame 1 (B, C, P, P)
    - patch2: patch from frame 2 (B, C, P, P)
    - prior_flow: Flor prior (B, 2)

    Output:
    - residual_flow: Predicted residual flow (B, 2)
    """
    patch_size: int
    channels: int
    num_features_conv1: int
    num_features_conv2: int
    mlp_hidden_size: int

    conv1x1: nnx.Conv
    bn1: nnx.BatchNorm
    convPxP: nnx.Conv
    bn2: nnx.BatchNorm
    mlp_hidden: nnx.Linear
    mlp_output: nnx.Linear

    def __init__(self,
                 patch_size: int,
                 channels: int,
                 rngs: nnx.Rngs,
                 *,
                 num_features_conv1: int = 8,
                 num_features_conv2: int = 16,
                 mlp_hidden_size: int = 16):
        """
            P: Patch size
            C: Number of channels from Convolution.
            rngs: A dictionary of JAX RNGs for parameter initialization.
            num_features_conv1: Output channels for the 1x1 convolution.
            num_features_conv2: Output channels for the PxP convolution (also MLP input size).
            mlp_hidden_size: Hidden layer size for the MLP.
        """
        # Store configuration parameters
        self.patch_size = patch_size
        self.channels = channels
        self.num_features_conv1 = num_features_conv1
        self.num_features_conv2 = num_features_conv2
        self.mlp_hidden_size = mlp_hidden_size

        # Conv1: cross-channel mixing (1x1 kernel)
        # `in_features` must be explicitly specified for nnx.Conv
        self.conv1x1 = nnx.Conv(
            in_features=(2 * channels + 2),  # (patch1_C + patch2_C + prior_flow_channels)
            out_features=self.num_features_conv1,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(
            num_features=self.num_features_conv1,
            use_running_average=True,  # Will be controlled by `train` flag in `__call__`
            rngs=rngs,
        )

        # Conv2: spatial mixing (PxP kernel)
        self.convPxP = nnx.Conv(
            in_features=self.num_features_conv1,  # Output channels from previous conv
            out_features=self.num_features_conv2,
            kernel_size=(patch_size, patch_size),
            padding='VALID',  # Crucial to reduce spatial dims to 1x1 for P=2
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(
            num_features=self.num_features_conv2,
            use_running_average=True,  # Will be controlled by `train` flag in `__call__`
            rngs=rngs,
        )

        # MLP Tail
        self.mlp_hidden = nnx.Linear(
            in_features=self.num_features_conv2,  # Input from flattened conv output
            out_features=self.mlp_hidden_size,
            rngs=rngs,
        )

        output_features = 2
        self.mlp_output = nnx.Linear(
            in_features=self.mlp_hidden_size,
            out_features=output_features,
            rngs=rngs,
        )

    def __call__(self, patch1: jax.Array, patch2: jax.Array, prior_flow: jax.Array,
                 train: bool = False):
        # Input shapes:
        # patch1: (B, C, P, P)
        # patch2: (B, C, P, P)
        # prior_flow: (B, 2)

        PB, _ = prior_flow.shape

        prior_as_patches = jnp.broadcast_to(prior_flow[:, :, None, None],
                                            (PB, 2, self.patch_size, self.patch_size))

        x = jnp.concatenate([patch1, patch2, prior_as_patches], axis=1)
        x_transposed = jnp.transpose(x, (0, 2, 3, 1))  # Conv expects NHWC order

        # x.shape: [B, 2*C+2, P, P]

        # 2. Convolutional Head
        x = self.conv1x1(x_transposed)
        # For nnx.BatchNorm, `use_running_average` is passed directly to the `__call__` method.
        # If `train=True`, batch stats are updated; if `train=False`, running averages are used.
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)  # Shape: (1, P, P, num_features_conv1) -> e.g., (1, 2, 2, 8)

        x = self.convPxP(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)  # Shape: (1, 1, 1, num_features_conv2) -> e.g., (1, 1, 1, 16)

        # Remove dummy batch dimension and flatten for MLP
        # Squeezing removes all 1-sized dimensions. Result: (num_features_conv2,)
        x = x.squeeze((1, 2))  # Shape: (num_features_conv2,) -> e.g., (16,)

        # 3. MLP Tail
        x = self.mlp_hidden(x)
        x = nnx.relu(x)  # Shape: (mlp_hidden_size,) -> e.g., (16,)

        x = self.mlp_output(x)  # Shape: (output_features,) -> e.g., (2,)

        return x


class ImageDecomposition(nnx.Module):

    def __init__(self, levels: int, rngs: nnx.Rngs, wavelet: str = 'db2'):
        self._levels = levels
        self._rngs = rngs
        self._wavelet = wavelet
        py_wavelet = pywt.Wavelet(wavelet)
        kernels = self._make_kernels(
            jnp.array(py_wavelet.dec_lo), jnp.array(py_wavelet.dec_hi)
        )
        self._kernels = nnx.Variable(kernels, collection='constants')


    def __call__(self, img: jnp.ndarray) -> list[jax.Array]:
        current = img
        pyramid = []
        for _ in range(self._levels):
            padded = jnp.pad(current, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge')
            decomposed = jax.lax.conv_general_dilated(
                padded,
                self._kernels.value,
                window_strides=(2, 2),
                padding='valid',
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            )
            current = decomposed[:, :1]
            pyramid.append(decomposed)

        return pyramid

    def _make_kernels(self, lo, hi):
        ll = jnp.outer(lo, lo)
        lh = jnp.outer(hi, lo)
        hl = jnp.outer(lo, hi)
        hh = jnp.outer(hi, hi)
        filters = jnp.stack([ll, lh, hl, hh], 0)
        filters = jnp.expand_dims(filters, 1)
        return filters


class PatchExtractor:
    """
    A module for extracting patches from images, handling both
    non-overlapping and shifted patch extraction.
    """
    _patch_size: int

    def __init__(self, patch_size: int):
        # rngs is needed for nnx.Module consistency, even if no params/state
        self._patch_size = patch_size

    def extract_non_overlapping_patches(self, images: jax.Array) -> jax.Array:
        """
        Extracts non-overlapping PxP patches from a batch of images.
        Input images: (B, C, H, W)
        Output: (B * num_patches, C, P, P)
        """
        B, C, H, W = images.shape
        P = self._patch_size
        if H % P != 0 or W % P != 0:
            raise ValueError("Image dimensions must be multiples of patch_size.")

        num_patches_H = H // P
        num_patches_W = W // P

        # (B, C, H, W) -> (B, C, H_patches, P, W_patches, P)
        reshaped_images = images.reshape(B, C, num_patches_H, P, num_patches_W, P)
        # (B, C, H_patches, P, W_patches, P) -> (B, H_patches, W_patches, C, P, P)
        permuted_images = reshaped_images.transpose(0, 2, 4, 1, 3, 5)
        # (B, H_patches, W_patches, C, P, P) -> (B * H_patches * W_patches, C, P, P)
        final_patches = permuted_images.reshape(B * num_patches_H * num_patches_W, C, P, P)
        return final_patches

    def extract_warped_patches_single_channel(self, frame: jax.Array, flow: jax.Array) -> jax.Array:
        warped = self._warp_patches_single_channel(frame, flow)
        return self.extract_non_overlapping_patches(warped[:, None, :, :]).squeeze(1)

    def _warp_patches_single_channel(self, frame: jax.Array, flow: jax.Array):
        """Extracts patches from frame shifted by "flow" (float).
        frame.shape: (B, H, W) (single channel)
        flow.shape (B, 2, H//self._patch_size, W//self._patch_size)
        """
        B, H, W = frame.shape
        FB, F, PH, PW = flow.shape
        assert FB == B
        assert F == 2

        batched_warp_patches = jax.vmap(self._warp_patches_single_channel_no_batch, in_axes=(0, 0))
        warped_patches = batched_warp_patches(frame, flow)
        return warped_patches

    def _warp_patches_single_channel_no_batch(self, frame: jax.Array, flow: jax.Array):
        H, W = frame.shape
        P = self._patch_size
        F, PH, PW = flow.shape
        assert F == 2
        assert H // P == PH
        assert W // P == PW

        grid_y, grid_x = jnp.meshgrid(
            jnp.arange(H),
            jnp.arange(W),
            indexing='ij'
        )
        grid_yx = jnp.stack([grid_y, grid_x], axis=0)
        patch_flow = flow * P
        flow_coords = patch_flow.repeat(P, axis=1).repeat(P, axis=2)
        map_coords = grid_yx + flow_coords
        warped_patches = map_coordinates(frame, map_coords, order=1)
        return warped_patches

    def extract_shifted_patches_multi_channel(self, frame: jax.Array,
                                              int_flows: jax.Array) -> jax.Array:
        """
        Extracts patches from `frame` shifted by `int_flows`.
        frame.shape: (B, H, W)
        int_flows.shape: (B, 2, H_level_patches, W_level_patches)
        Output: (B * num_patches, C, P, P)
        """
        B, C, H, W = frame.shape
        P = self._patch_size

        num_patches_H = H // P
        num_patches_W = W // P
        grid_y_coords, grid_x_coords = jnp.meshgrid(jnp.arange(num_patches_H),
                                                    jnp.arange(num_patches_W),
                                                    indexing='ij')
        grid_y_offsets = grid_y_coords * P
        grid_x_offsets = grid_x_coords * P

        def _extract_shifted_patches_from_single_frame(single_frame, single_int_flow):
            int_flows_y = single_int_flow[0]
            int_flows_x = single_int_flow[1]
            start_y_all_patches_single = grid_y_offsets + int_flows_y
            start_x_all_patches_single = grid_x_offsets + int_flows_x
            start_coords_flattened_single = jnp.stack(
                [start_y_all_patches_single.ravel(), start_x_all_patches_single.ravel()],
                axis=-1)

            def _extract_one_patch(start_yx):
                start_y, start_x = start_yx[0], start_yx[1]
                return jax.lax.dynamic_slice(
                    single_frame,
                    (0, start_y, start_x),
                    (C, P, P)
                )

            patches_for_single_frame = jax.vmap(_extract_one_patch, in_axes=0, out_axes=0)(
                start_coords_flattened_single)
            return patches_for_single_frame

        shifted_patches_batch = jax.vmap(
            _extract_shifted_patches_from_single_frame, in_axes=(0, 0), out_axes=0)(
            frame, int_flows)

        return shifted_patches_batch.reshape(B * num_patches_H * num_patches_W, C, P, P)


class SingleLayerFlowPredictor(nnx.Module):
    def __init__(self, patch_size: int, channels: int, *, rngs: nnx.Rngs):
        self._patch_size = patch_size
        self._rngs = rngs
        self._patch_extractor = PatchExtractor(patch_size=patch_size)
        self._flow_predictor_single_patch = PatchFlowPredictor(
            patch_size=patch_size, channels=channels, rngs=rngs
        )

    def __call__(self,
                 frame1: jax.Array,
                 frame2: jax.Array,
                 prior: jax.Array, train: bool) -> jax.Array:
        """
        frame1 and frame2 have shapes: [B, C, H_level, W_level]
        prior has shape [B, 2, H_level_patches, W_level_patches]
        """
        B, _, H_level, W_level = frame1.shape
        P = self._patch_size
        PH = H_level // P
        PW = W_level // P
        assert prior.shape == (B, 2, PH, PW), \
            f"Prior shape mismatch: Expected ({B}, 2, {PH}, {PW}), got {prior.shape}"
        whole_priors = jnp.round(prior).astype(jnp.int32)
        remainder_priors = prior - whole_priors
        frame2_patches_flat = self._patch_extractor.extract_shifted_patches_multi_channel(
            frame2, whole_priors
        )
        frame1_patches_flat = self._patch_extractor.extract_non_overlapping_patches(frame1)
        num_patches_total = B * PH * PW
        flattened_patch_priors_flat = remainder_priors.transpose(0, 2, 3, 1).reshape(
            num_patches_total, 2)

        residual_flow_flat = self._flow_predictor_single_patch(
            frame1_patches_flat,
            frame2_patches_flat,
            flattened_patch_priors_flat,
            train=train
        )
        # Reshape back to (B, 2, PH, PW) for residual flow per patch grid.
        # The output of vmap is (B*PH*PW, 2). Reshape to (B, PH, PW, 2) then transpose.
        residual_flow_patches = residual_flow_flat.reshape(B, PH, PW, 2).transpose(
            0, 3, 1, 2)  # (B, 2, PH, PW)

        # Add residual flow to the prior (which is already (B, 2, PH, PW))
        net_flow = residual_flow_patches + prior

        return net_flow


class MultiLayerFlowPredictor(nnx.Module):

    def __init__(self, patch_size: int, channels: int, *, rngs: nnx.Rngs):
        smoothing_kernel = jnp.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=jnp.float32).reshape(1, 1, 3, 3)
        norm_smoothing_kernel = smoothing_kernel / jnp.sum(smoothing_kernel)
        self._smoothing_kernel = nnx.Variable(norm_smoothing_kernel, collection='constants')
        self._patch_size = patch_size
        self._single_layer_flow_predictor = SingleLayerFlowPredictor(
            patch_size, channels, rngs=rngs
        )

    def __call__(self,
                 layers1: Sequence[jax.Array],
                 layers2: Sequence[jax.Array],
                 prior: jax.Array, train: bool) -> Sequence[jax.Array]:
        level_flows = []
        for f1, f2 in reversed(list(zip(layers1, layers2))):
            level_flow = self._single_layer_flow_predictor(f1, f2, prior, train=train)
            level_flows.append(level_flow)
            prior = self.expand_flow_to_next_level_prior(level_flow, smooth=True)
        return level_flows[::-1]

    def _smoothen(self, a: jax.Array) -> jax.Array:
        """
        Smoothen entries on an array(img) by interpolating each entry with their neighbors with a
        specific kernel.
        The borders are copied over so that the kernel has enough context for smoothing.
        """
        B, F, H, W = a.shape  # C_flow is 2 (for dx, dy)
        assert F == 2
        reshaped_for_conv = a.reshape(B * F, H, W)[:, jnp.newaxis, :, :]

        extra_borders = jnp.pad(reshaped_for_conv,
                                ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge')
        smoothed_reshaped = jax.lax.conv(extra_borders, self._smoothing_kernel.value, (1, 1), 'valid')
        smoothed = smoothed_reshaped.reshape(B, F, H, W)
        return smoothed

    def expand_flow_to_next_level_prior(self, level_flow: jax.Array, smooth: bool) -> jax.Array:
        upscaled_flow = 2 * level_flow
        expanded_upscaled_flow = upscaled_flow.repeat(2, axis=2).repeat(2, axis=3)
        if smooth:
            expanded_upscaled_flow = self._smoothen(expanded_upscaled_flow)
        return expanded_upscaled_flow


class SingleLevelPhotometricLoss(nnx.Module):
    """
    Calculates photometric consistency loss (1 - NCC) for a single pair of images
    and their corresponding flow field, using 4x4 patch-wise NCC.
    """
    _ncc_patch_size: int  # Window size for NCC calculation (e.g., 4)
    _patch_extractor: PatchExtractor  # Plain class instance

    def __init__(self, ncc_patch_size: int, *, rngs: nnx.Rngs):
        self._ncc_patch_size = ncc_patch_size
        # Instantiate the plain PatchExtractor class
        self._patch_extractor = PatchExtractor(patch_size=ncc_patch_size)

    def __call__(self, img1: jax.Array, img2: jax.Array, flow: jax.Array) -> jax.Array:
        B, H, W = img1.shape
        _, _, FH, FW = flow.shape
        P = self._ncc_patch_size
        assert H // FH == P
        assert W // FW == P

        frame1_patches = self._patch_extractor.extract_non_overlapping_patches(
            img1[:, None, :, :]  # (B, 1, H, W)
        ).squeeze(1)
        frame2_patches = self._patch_extractor.extract_warped_patches_single_channel(
            img2, flow
        )
        # Both frame patches shapes should be: (B*FH*FW, P, P)

        # Vmap the NCC calculation over the flattened patches
        # Input patches are (B * num_patches, 1, P, P)
        ncc_scores_flat = jax.vmap(self._calculate_single_ncc, in_axes=(0, 0))(
            frame1_patches, frame2_patches
        )  # Output shape: (B * PH * PW)
        # Compute loss (1 - NCC) and sum over all patches
        per_patch_loss = (1.0 - ncc_scores_flat) / 2.0  # Scale to [0,1]
        per_batch_patches_loss = per_patch_loss.reshape(B, FH, FW)

        # --- CHANGE: Return MEAN over patches per batch item ---
        level_loss = jnp.mean(per_batch_patches_loss, axis=(1, 2))
        # --- END CHANGE ---

        return level_loss

    def _calculate_single_ncc(self, patch1: jax.Array, patch2: jax.Array) -> jax.Array:
        """Calculates NCC between two single patches of shape (1, P, P)."""
        mean1 = jnp.mean(patch1)
        mean2 = jnp.mean(patch2)
        dev1 = patch1 - mean1
        dev2 = patch2 - mean2
        numerator = jnp.sum(dev1 * dev2)
        sum_sq_dev1 = jnp.sum(dev1 ** 2)
        sum_sq_dev2 = jnp.sum(dev2 ** 2)

        # Add epsilon inside each square root term (Numerical stability fix)
        stddev1 = jnp.sqrt(sum_sq_dev1 + 1e-6)
        stddev2 = jnp.sqrt(sum_sq_dev2 + 1e-6)
        denominator = stddev1 * stddev2 + 1e-6  # Keep epsilon for the final division just in case

        ncc_score = numerator / denominator
        ncc_score = jnp.clip(ncc_score, -1.0, 1.0)
        return ncc_score


class MultiLevelPhotometricLoss(nnx.Module):
    _alpha: float
    _beta: float
    _gamma: float

    def __init__(self, ncc_patch_size: int, *, rngs: nnx.Rngs,
                 alpha: float = 10.0, beta: float = 1.0, gamma: float = 1.0):
        self._single_level_photometric_loss = SingleLevelPhotometricLoss(
            ncc_patch_size=ncc_patch_size,
            rngs=rngs
        )
        self._patch_size = ncc_patch_size
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    # --- CHANGE: Update return type annotation to include auxiliary data dict ---
    def __call__(self, pyramid1: Sequence[jax.Array], pyramid2: Sequence[jax.Array],
                 flow: Sequence[jax.Array]) -> tuple[
        jax.Array, dict[str, jax.Array | Sequence[jax.Array]]]:
        # --- END CHANGE ---

        # Collect MEAN losses per level (each shape (B,))
        mean_level_losses_per_batch_item = [
            self._single_level_photometric_loss(f1, f2, flow)
            for f1, f2, flow in zip(pyramid1, pyramid2, flow)
        ]

        # Stack the (B,) arrays from each level to get (Num_Levels, B)
        stacked_mean_losses = jnp.stack(mean_level_losses_per_batch_item, axis=0)  # (N_Levels, B)

        # --- NEW: Calculate average UNWEIGHTED loss per level across the batch ---
        average_unweighted_losses_per_level = jnp.mean(stacked_mean_losses, axis=1)  # (N_Levels,)
        # --- END NEW ---

        # Implement loss weighting
        weights = []
        num_levels = len(mean_level_losses_per_batch_item)

        for i in range(num_levels - 1):
            # weights[i] applies to mean_level_losses_per_batch_item[i] (finer level)
            # derived from mean_level_losses_per_batch_item[i+1] (coarser level)
            weight_i = jax.nn.sigmoid(
                self._beta - self._alpha * stacked_mean_losses[i + 1, :])  # (B,)
            weights.append(weight_i)

        # The coarsest level (index num_levels - 1) gets a fixed weight gamma
        coarsest_level_weight = jnp.full_like(stacked_mean_losses[num_levels - 1, :],
                                              self._gamma)  # (B,)
        weights.append(coarsest_level_weight)

        # Stack weights to (Num_Levels, B)
        stacked_weights = jnp.stack(weights, axis=0)  # (N_Levels, B)

        # Compute total weighted loss per batch item: sum(weight * mean_loss) over levels
        total_loss_per_batch_item = jnp.sum(stacked_weights * stacked_mean_losses, axis=0)  # (B,)

        # Average across the batch to get a single scalar loss for the entire batch
        final_scalar_loss = jnp.mean(total_loss_per_batch_item, axis=0)  # Scalar

        # --- CHANGE: Return scalar loss and auxiliary data dict ---
        aux_data = {
            'weights': weights,  # List of (B,) arrays, ordered finest to coarsest
            'mean_unweighted_losses_per_level': average_unweighted_losses_per_level
            # (N_Levels,) array
        }
        return final_scalar_loss, aux_data


class ImagePairFlowPredictor(nnx.Module):
    def __init__(self, patch_size: int, channels: int,
                 levels: int, wavelet: str, ncc_patch_size: int, rngs: nnx.Rngs,
                 loss_alpha: float = 10.0, loss_beta: float = 1.0, loss_gamma: float = 1.0):
        self._layers_predictor = MultiLayerFlowPredictor(
            patch_size=patch_size,
            channels=channels,
            rngs=rngs
        )
        self._image_decomposition = ImageDecomposition(levels=levels, rngs=rngs, wavelet=wavelet)
        self._loss = MultiLevelPhotometricLoss(ncc_patch_size=ncc_patch_size, rngs=rngs,
                                               alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma)
        self.rngs = rngs
        self._patch_size = patch_size

    # --- CHANGE: Update return type annotation for loss and aux ---
    def __call__(self, f1: jax.Array, f2: jax.Array, priors: jax.Array, train: bool
                 ) -> tuple[
        Sequence[jax.Array], Sequence[jax.Array], Sequence[jax.Array], jax.Array, dict
    ]:

        # --- END CHANGE ---
        pyramid1 = self._image_decomposition(f1)
        pyramid2 = self._image_decomposition(f2)
        flow_pyramid = self._layers_predictor(pyramid1, pyramid2, priors, train=train)
        loss_pyramid1 = self._get_single_channel_fine_grained_pyramid(f1, pyramid1)
        loss_pyramid2 = self._get_single_channel_fine_grained_pyramid(f2, pyramid2)

        # --- CHANGE: Unpack loss and auxiliary data dict ---
        loss, aux_data = self._loss(loss_pyramid1, loss_pyramid2, flow_pyramid)
        # --- END CHANGE ---

        # --- CHANGE: Return auxiliary data dict ---
        return pyramid1, pyramid2, flow_pyramid, loss, aux_data


    def _get_single_channel_fine_grained_pyramid(self, original_frame: jax.Array,
                                                 dwt_pyramid_raw: Sequence[jax.Array]) -> Sequence[
        jax.Array]:
        """
        Constructs the sequence of single-channel images for photometric loss calculation,
        including the original image and subsequent LL channels from DWT levels.

        Args:
            original_frame: The initial input image (B, 1, H, W).
            dwt_pyramid_raw: The full DWT pyramid output from ImageDecomposition,
                             ordered finest to coarsest: [L0, L1, L2, L3, L4].

        Returns:
            A sequence of single-channel images (B, H_level, W_level), ordered
            from finest (original) to progressively coarser DWT levels:
            [Original_image_LL_channel, L0_LL, L1_LL, L2_LL, L3_LL].
            Note: The coarsest DWT level (L4_LL) is excluded to match the length
            of the flow_pyramid (which has 'levels' elements, 0 to levels-1).
        """
        images_for_loss = [
            original_frame[:, 0, :, :]]  # Start with the original image's single channel (B, H, W)

        # Append the LL channel from each DWT level, excluding the very coarsest one.
        # This loop iterates 'levels - 1' times (e.g., if levels=5, then 4 times for L0, L1, L2, L3).
        # dwt_pyramid_raw[i] will be the DWT level for L_i.
        for i in range(
                len(dwt_pyramid_raw) - 1):  # If dwt_pyramid_raw has 'levels' elements (L0..L_levels-1)
            # we iterate from i=0 to i=levels-2.
            images_for_loss.append(dwt_pyramid_raw[i][:, 0, :, :])  # Append LL channel of L_i

        # Resulting list length: 1 (original image) + (levels - 1) (DWT LL levels) = levels.
        # This matches the length of 'flow_pyramid'.
        return images_for_loss


## aux
def load_image_to_array(img_path: Path) -> jax.Array:
    img = Image.open(img_path)
    as_array = jnp.asarray(img)
    assert as_array.ndim == 2
    return as_array / 255.0


@jax.jit
def run():
    f11 = load_image_to_array(Path('./datasets/frames/bookshelf/frame_000200.png'))
    f21 = load_image_to_array(Path('./datasets/frames/bookshelf/frame_000205.png'))
    f12 = load_image_to_array(Path('./datasets/frames/bookshelf/frame_000300.png'))
    f22 = load_image_to_array(Path('./datasets/frames/bookshelf/frame_000305.png'))
    # Use explicit RNG keys for clarity
    rngs = flax.nnx.Rngs(params=41, prior=jax.random.PRNGKey(41))

    f1_batched = jnp.stack([f11, f12], axis=0)[:,None,:,:]
    f2_batched = jnp.stack([f21, f22], axis=0)[:,None,:,:]

    # Generate dummy priors consistent with the model's expected shape for the coarsest level
    # Use the same parameters as in train.py for consistency.
    B, _, H, W = f1_batched.shape
    levels_in_run = 5 # Should match ImagePairFlowPredictor's levels
    patch_size_in_run = 2 # Should match ImagePairFlowPredictor's patch_size
    PH = H // (2 ** levels_in_run) // patch_size_in_run
    PW = W // (2 ** levels_in_run) // patch_size_in_run
    dummy_priors = jax.random.normal(rngs.prior(), (B, 2, PH, PW)) * 0.01

    model = ImagePairFlowPredictor(patch_size=patch_size_in_run, channels=4, levels=levels_in_run, wavelet='db2',
                                   ncc_patch_size=4,rngs=rngs)
    # Pass the dummy prior
    return model(f1_batched, f2_batched, priors=dummy_priors, train=False)

if __name__ == '__main__':
    pyramid1, pyramid2, flow_pyramid, loss = run()
    print("Flow pyramid shapes (finest to coarsest):", [p.shape for p in flow_pyramid])
    print("Total Loss (scalar):", loss)
