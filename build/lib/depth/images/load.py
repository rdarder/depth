import jax
import jax.numpy as jnp
import cv2


def get_offsets(input_size: int, output_size: int):
    reminder = input_size - output_size
    start = reminder // 2
    end = input_size - (reminder - start)
    return start, end


def crop(img: jax.Array, size: int) -> jax.Array:
    H, W = img.shape
    start_y, end_y = get_offsets(H, size)
    start_x, end_x = get_offsets(W, size)
    return img[start_y:end_y, start_x:end_x]


def load_frame_from_path(frame_path: str, crop_size: int):
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f'frame not found at {frame_path}')
    as_array = jnp.asarray(img)
    assert as_array.ndim == 2
    cropped = crop(as_array, crop_size)
    normalized = cropped / 255.0
    with_channel_dimension = normalized[None, :, :]
    return with_channel_dimension
