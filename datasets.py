import glob
import os
import re  # For sorting frame numbers naturally

import tensorflow as tf  # For tf.data and tf.image operations

# --- Configuration ---
DATASET_DIR = "datasets/frames"  # Your root dataset directory
IMAGE_EXTENSION = ".png"
TARGET_IMG_HEIGHT = 128  # Example, adjust to your needs
TARGET_IMG_WIDTH = 128  # Example, adjust to your needs
SHUFFLE_BUFFER_SIZE = 50_000  # Adjust based on dataset size


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    """Key for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def get_consecutive_frame_pairs(
    dataset_root: str, image_extension: str
) -> list[tuple[str, str]]:
    """
    Scans the dataset directory to find pairs of consecutive frames.

    Args:
        dataset_root: Path to the root directory containing video subfolders.
        image_extension: The file extension of the frame images (e.g., ".png").

    Returns:
        A list of tuples, where each tuple is (path_to_frame_n, path_to_frame_n_plus_1).
    """
    all_frame_pairs = []
    video_folders = [
        os.path.join(dataset_root, d)
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    for video_folder in video_folders:
        frame_files = glob.glob(os.path.join(video_folder, f"frame_*{image_extension}"))
        frame_files.sort(key=natural_sort_key)  # Sort frames numerically

        if len(frame_files) < 2:
            print(
                f"Warning: Not enough frames in {video_folder} to form pairs. Skipping."
            )
            continue
        for skip in range(1, 8):
            for i in range(len(frame_files) - skip):
                frame_n_path = frame_files[i]
                frame_skip_ahead = frame_files[i + skip]
                all_frame_pairs.append((frame_n_path, frame_skip_ahead))
                all_frame_pairs.append((frame_skip_ahead, frame_n_path))

    print(f"Found {len(all_frame_pairs)} consecutive frame pairs.")
    return all_frame_pairs


def load_and_preprocess_image_pair(
    path_t: tf.Tensor, path_t_plus_1: tf.Tensor, target_height: int, target_width: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Loads a single pair of images from their paths, decodes, resizes, and normalizes.
    """

    def _load_image(path_tensor):
        img_raw = tf.io.read_file(path_tensor)
        img = tf.image.decode_png(img_raw, channels=1)  # Grayscale
        img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0,1]
        img = tf.image.resize(
            img, [target_height, target_width], method=tf.image.ResizeMethod.BILINEAR
        )  # or NEAREST
        return img

    img_t = _load_image(path_t)
    img_t_plus_1 = _load_image(path_t_plus_1)

    return img_t, img_t_plus_1


def create_dataset(
    frame_pairs_list: list[tuple[str, str]],
    target_height: int,
    target_width: int,
    batch_size: int,
    shuffle_buffer_size: int,
    is_training: bool = True,
) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset for optical flow training.
    """
    path_t_list, path_t_plus_1_list = zip(*frame_pairs_list)

    dataset = tf.data.Dataset.from_tensor_slices(
        (list(path_t_list), list(path_t_plus_1_list))
    )

    if is_training:
        dataset = dataset.shuffle(shuffle_buffer_size)

    # Load and preprocess images
    dataset = dataset.map(
        lambda path_t, path_t_plus_1: load_and_preprocess_image_pair(
            path_t, path_t_plus_1, target_height, target_width
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # TODO: Add augmentations here if needed (e.g., random flip, time reversal)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# if __name__ == "__main__":
#     # --- Create dummy dataset for testing ---
#     dummy_dataset_root = "datasets_dummy"
#     if not os.path.exists(dummy_dataset_root):
#         os.makedirs(dummy_dataset_root)
#
#     for video_name in ["video1", "video2"]:
#         video_path = os.path.join(dummy_dataset_root, video_name)
#         if not os.path.exists(video_path):
#             os.makedirs(video_path)
#         for i in range(1, 6):  # 5 frames per dummy video
#             # Create a simple PNG file (e.g., a small gradient)
#             try:
#                 # For this test, we'll create actual small PNGs if tf can write them easily
#                 # or just skip if it's too much hassle for a dummy script.
#                 # For now, let's assume they exist or are created by your existing scripts.
#                 # We'll just create empty files to test path finding.
#                 dummy_frame_filename = os.path.join(
#                     video_path, f"frame_{i:06d}{IMAGE_EXTENSION}"
#                 )
#                 if not os.path.exists(dummy_frame_filename):
#                     # Create a tiny dummy PNG using tf.image.encode_png
#                     dummy_image_tensor = tf.ones((10, 10, 1), dtype=tf.uint8) * (
#                         i * 10
#                     )  # Simple gradient
#                     dummy_image_encoded = tf.image.encode_png(dummy_image_tensor)
#                     tf.io.write_file(dummy_frame_filename, dummy_image_encoded)
#                     # print(f"Created dummy file: {dummy_frame_filename}")
#
#             except Exception as e:
#                 print(
#                     f"Could not create dummy PNG {dummy_frame_filename}: {e}. "
#                     "Please ensure you have a 'datasets' folder with PNGs for testing."
#                 )
#                 # If PNG creation fails, the glob will just find fewer/no files.
#     # --- End dummy dataset creation ---
#
#     # 1. Get frame pairs
#     frame_pairs = get_consecutive_frame_pairs(
#         dummy_dataset_root, IMAGE_EXTENSION
#     )  # Use dummy for test
#     # frame_pairs = get_consecutive_frame_pairs(DATASET_DIR, IMAGE_EXTENSION) # Use this for your actual data
#
#     if not frame_pairs:
#         print("No frame pairs found. Exiting test.")
#     else:
#         print(f"\nExample frame pair: {frame_pairs[0]}")
#
#         # 2. Create the tf.data.Dataset
#         train_dataset = create_dataset(
#             frame_pairs,
#             target_height=TARGET_IMG_HEIGHT,
#             target_width=TARGET_IMG_WIDTH,
#             batch_size=BATCH_SIZE,
#             shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
#             is_training=True,
#         )
#
#         print(f"\nDataset spec: {train_dataset.element_spec}")
#
#         # 3. Iterate over a few batches (as JAX would)
#         num_batches_to_show = 2
#         for i, (frame_t_batch, frame_t_plus_1_batch) in enumerate(
#             train_dataset.take(num_batches_to_show)
#         ):
#             print(f"\nBatch {i + 1}:")
#             print(
#                 f"  Frame T batch shape: {frame_t_batch.shape}, dtype: {frame_t_batch.dtype}"
#             )
#             print(
#                 f"  Frame T+1 batch shape: {frame_t_plus_1_batch.shape}, dtype: {frame_t_plus_1_batch.dtype}"
#             )
#             # In JAX training loop, you'd convert these to JAX arrays:
#             # jax_frame_t_batch = jnp.array(frame_t_batch.numpy()) # or similar if using as_numpy_iterator
#
#             if i == 0 and BATCH_SIZE > 0:  # Check values for one image
#                 print(
#                     f"  Example pixel value from frame_t (batch 0, item 0, pixel 0,0): {frame_t_batch[0, 0, 0, 0]}"
#                 )
#                 print(f"  Max value in frame_t_batch: {tf.reduce_max(frame_t_batch)}")
#                 print(f"  Min value in frame_t_batch: {tf.reduce_min(frame_t_batch)}")
#
#         # Example of how to get a JAX-compatible iterator
#         print("\nGetting a JAX-compatible iterator (as_numpy_iterator):")
#         numpy_iterator = train_dataset.as_numpy_iterator()
#         try:
#             np_frame_t_batch, np_frame_t_plus_1_batch = next(numpy_iterator)
#             print(
#                 f"  NumPy Frame T batch shape: {np_frame_t_batch.shape}, type: {type(np_frame_t_batch)}"
#             )
#             print(
#                 f"  NumPy Frame T+1 batch shape: {np_frame_t_plus_1_batch.shape}, type: {type(np_frame_t_plus_1_batch)}"
#             )
#         except StopIteration:
#             print("  Dataset was empty or too small for one batch.")
#
#     # Clean up dummy dataset (optional)
#     # import shutil
#     # if os.path.exists(dummy_dataset_root):
#     #     shutil.rmtree(dummy_dataset_root)
#     #     print(f"\nCleaned up {dummy_dataset_root}")
