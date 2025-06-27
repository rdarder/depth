from jax import numpy as jnp

from depth.images.separable_convolution import conv_output_size_steps, conv_output_size
from depth.model.settings import ModelSettings


def generate_zero_priors(batch_size, settings: ModelSettings):
    # pyramid decomposition
    decomposed_patches = conv_output_size_steps(settings.img_size, settings.decompose_kernel_size,
                                                settings.decompose_stride, settings.levels)
    extracted_patches = conv_output_size(decomposed_patches, settings.patch_size,
                                         settings.patch_stride)
    initial_priors = jnp.zeros((batch_size, extracted_patches, extracted_patches, 2),
                               dtype=jnp.float32)
    return initial_priors
