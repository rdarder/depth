import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

# Assuming the prepared data is in the 'processed_data' directory
PROCESSED_DATA_DIR = "datasets/frames"
TARGET_IMAGE_SIZE = (64, 64)  # HxW


def load_and_pair_frames(
    data_dir: str = PROCESSED_DATA_DIR, image_size: tuple = TARGET_IMAGE_SIZE
) -> jax.Array:
    """
    Loads prepared grayscale images from subdirectories, pairs consecutive frames,
    and returns them as a single JAX array.

    Args:
        data_dir: The root directory containing video subdirectories with frames.
        image_size: The expected HxW size of the square images.

    Returns:
        A JAX array of shape (total_pairs, 2, H, W, 1) containing frame pairs.
        Returns None if no valid pairs are found.
    """
    all_pairs = []
    video_dirs = sorted(
        glob.glob(os.path.join(data_dir, "*"))
    )  # Get video subdirectories

    if not video_dirs:
        print(f"Warning: No video subdirectories found in {data_dir}.")
        return None

    for video_dir in video_dirs:
        if not os.path.isdir(video_dir):
            continue  # Skip if it's not a directory

        print(f"Loading frames from: {video_dir}")
        # List and sort frame files numerically
        frame_files = sorted(
            glob.glob(os.path.join(video_dir, "frame_*.png")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1]),
        )

        if len(frame_files) < 2:
            print(
                f"Warning: Not enough frames ({len(frame_files)}) in {video_dir} to form pairs. Skipping."
            )
            continue

        # Load all frames for this video
        frames = []
        for frame_file in frame_files:
            try:
                # Open image, convert to grayscale (should already be, but good practice)
                # Ensure it's in L mode (8-bit grayscale)
                img = Image.open(frame_file).convert("L")

                # Check size
                if img.size != image_size:
                    print(
                        f"Warning: Skipping frame {frame_file} due to unexpected size {img.size}. Expected {image_size}"
                    )
                    continue

                # Convert to numpy, add channel dimension (H, W) -> (H, W, 1)
                frame_np = np.array(img, dtype=np.float32)[:, :, np.newaxis]
                # Normalize pixel values to [0, 1] if they are 0-255 (common for PNG)
                if frame_np.max() > 1.0:
                    frame_np /= 255.0

                frames.append(frame_np)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}. Skipping.")
                continue  # Skip this frame

        if len(frames) < 2:
            print(
                f"Warning: Not enough *valid* loaded frames ({len(frames)}) in {video_dir} to form pairs. Skipping."
            )
            continue

        # Create consecutive pairs (frame_t, frame_{t+1})
        for skip_frames in range(1, 5):
            forward_pairs = [
                (frames[i], frames[i + skip_frames])
                for i in range(len(frames) - skip_frames)
            ]
            reverse_pairs = [
                (frames[i + skip_frames], frames[i])
                for i in range(len(frames) - skip_frames)
            ]
            video_pairs = forward_pairs + reverse_pairs
            all_pairs.extend(video_pairs)
            print(f"Added {len(video_pairs)} pairs from {video_dir}")

    if not all_pairs:
        print("No frame pairs were successfully loaded from any video directory.")
        return None

    # Stack all pairs into a single numpy array
    # Shape will be (total_pairs, 2, H, W, 1)
    all_pairs_np = np.stack([np.stack(pair, axis=0) for pair in all_pairs], axis=0)

    # Convert the entire dataset to a JAX array
    all_pairs_jax = jnp.asarray(all_pairs_np)

    print(f"\nSuccessfully loaded and paired {len(all_pairs)} frames.")
    print(f"Total dataset shape: {all_pairs_jax.shape}")

    return all_pairs_jax


def create_batches(
    dataset: jax.Array,
    batch_size: int,
    shuffle: bool = True,
    key: jax.random.PRNGKey = None,
):
    """
    Creates batches from the dataset.

    Args:
        dataset: A JAX array of shape (total_items, ...).
        batch_size: The size of each batch.
        shuffle: Whether to shuffle the dataset before batching.
        key: A JAX PRNGKey for shuffling.

    Yields:
        Batches of the dataset, each a JAX array of shape (batch_size, ...).
    """
    num_items = dataset.shape[0]
    indices = jnp.arange(num_items)

    if shuffle:
        if key is None:
            raise ValueError("A PRNGKey is required for shuffling.")
        indices = jax.random.permutation(key, indices)

    for i in range(0, num_items, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield dataset[batch_indices]


# --- Example Usage (assuming you ran the shell script first) ---
if __name__ == "__main__":
    # Need a JAX PRNG key for shuffling
    key = jax.random.PRNGKey(3)

    # Load the entire dataset into memory
    # Make sure you have run the shell script to generate processed_data first!
    frame_pairs_dataset = load_and_pair_frames()

    if frame_pairs_dataset is not None:
        # Example of iterating through batches
        batch_size = 8
        print(f"\nCreating batches of size {batch_size}...")

        # Split the key for shuffling
        shuffle_key, key = jax.random.split(key)

        num_batches = 0
        for i, batch in enumerate(
            create_batches(
                frame_pairs_dataset, batch_size, shuffle=True, key=shuffle_key
            )
        ):
            print(f"  Batch {i + 1} shape: {batch.shape}")
            # 'batch' has shape (batch_size, 2, H, W, 1)
            # batch[:, 0] is the first frame (image1_L0 equivalent for this batch)
            # batch[:, 1] is the second frame (image2_L0 equivalent for this batch)
            num_batches += 1

        print(f"Finished iterating through {num_batches} batches.")

    else:
        print(
            f"Dataset loading failed. Please check your {PROCESSED_DATA_DIR} directory and input videos."
        )
