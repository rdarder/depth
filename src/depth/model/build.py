from flax import nnx as nnx

from depth.model.multi_level_flow import PyramidFlowEstimator
from depth.model.patch_flow import PatchFlowEstimator
from depth.model.settings import ModelSettings
from depth.model.single_level_flow import LevelFlowEstimator


def make_model(seed: int, train: bool, settings: ModelSettings) -> PyramidFlowEstimator:
    rngs = nnx.Rngs(seed)
    patch_flow_estimator = PatchFlowEstimator(
        patch_size=settings.patch_size,
        num_channels=1,
        features=settings.predictor_features,
        mlp_hidden_size=settings.mlp_hidden_size,
        train=train,
        rngs=rngs
    )
    level_flow_estimator = LevelFlowEstimator(stride=settings.patch_stride,
                                              flow_estimator=patch_flow_estimator)
    pyramid_flow_estimator = PyramidFlowEstimator(level_flow_estimator)
    return pyramid_flow_estimator


