from importlib import resources

from flax import nnx
import jax
import jax.numpy as jnp

from depth.images.load import load_frame_from_path
from depth.images.separable_convolution import conv_output_size
from depth.model.multi_level_flow import PyramidFlowEstimator
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.pyramid import ImagePyramidDecomposer, ImageDownscaler
from depth.model.single_level_flow import LevelFlowEstimator


class FramePairFlow(nnx.Module):
    def __init__(self, estimator: PyramidFlowEstimator, decomposer: ImagePyramidDecomposer):
        self.estimator = estimator
        self.decomposer = decomposer

    def __call__(self, f1: jax.Array, f2: jax.Array, priors: jax.Array):
        """
        f1, f2: N, H, W, C
        priors: N, PY, PX, 2

        PY must match the amount of patches to get from the coarsest decomposition of the frame.
        """
        pyramid1 = self.decomposer(f1)
        pyramid2 = self.decomposer(f2)
        B, H, W, C = pyramid1[-1].shape
        coarsest_patches_y = conv_output_size(H, self.estimator.patch_size, self.estimator.stride)
        coarsest_patches_x = conv_output_size(W, self.estimator.patch_size, self.estimator.stride)
        assert priors.shape == (B, coarsest_patches_y, coarsest_patches_x, 2)
        return self.estimator(pyramid1, pyramid2, priors), pyramid1, pyramid2


def test_frame_pair_flow():
    frame1_path = resources.files('depth.test_fixtures') / "frame1.png"
    frame2_path = resources.files('depth.test_fixtures') / "frame2.png"
    frame1 = load_frame_from_path(str(frame1_path), 158)
    frame2 = load_frame_from_path(str(frame2_path), 158)
    batch1 = jnp.stack([frame1, frame2], axis=0)
    batch2 = jnp.stack([frame2, frame1], axis=0)
    rngs = nnx.Rngs(0)
    image_downscaler = ImageDownscaler(alpha=0.5, stride=2)
    pyramid_decomposer = ImagePyramidDecomposer(image_downscaler, levels=4)
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=4, num_channels=1, features_dim=8, train=False, rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=2, flow_estimator=patch_flow_estimator)
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator)
    frame_flow_estimator = FramePairFlow(pyramid_flow_estimator, pyramid_decomposer)
    prior = jnp.zeros((2, 3, 3, 2), jnp.float32)
    flow_with_loss_pyramid, _, _ = frame_flow_estimator(batch1, batch2, prior)
    jax.block_until_ready(flow_with_loss_pyramid)
    assert flow_with_loss_pyramid[-1].shape == (2, 3, 3, 3)
    assert flow_with_loss_pyramid[-2].shape == (2, 8, 8, 3)
    assert flow_with_loss_pyramid[-3].shape == (2, 18, 18, 3)
    assert flow_with_loss_pyramid[-4].shape == (2, 38, 38, 3)
    assert flow_with_loss_pyramid[-5].shape == (2, 78, 78, 3)
