"""Dataset loading, deterministic splitting, and tf.data preprocessing pipeline."""

from typing import List, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds


def get_augmentation_layer(seed: int = 42) -> tf.keras.Sequential:
    """Create a light data augmentation pipeline for training images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.1, seed=seed),
        ],
        name="data_augmentation",
    )


def load_malaria_datasets(
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], dict]:
    """Load TFDS Malaria dataset and build deterministic train/val/test tf.data pipelines.

    Images are returned as float32 tensors in [0, 255] so both Custom CNN
    (Rescaling 1/255) and MobileNetV2 (preprocess_input) can share the same
    cached dataset pipeline without duplicating RAM usage.

    Returns:
        train_ds, val_ds, test_ds, class_names, split_counts
    """
    full_ds, ds_info = tfds.load(
        "malaria",
        split="train",
        as_supervised=True,
        with_info=True,
    )

    class_names = list(ds_info.features["label"].names)
    total_examples = ds_info.splits["train"].num_examples

    train_size = int(total_examples * train_ratio)
    val_size = int(total_examples * val_ratio)
    test_size = total_examples - train_size - val_size

    def resize_uint8(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, img_size, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)
        return image, label

    # Resize to uniform shape as uint8 before deterministic full-dataset shuffle
    # to keep memory usage low while ensuring balanced class distribution across splits.
    resized_ds = full_ds.map(resize_uint8, num_parallel_calls=tf.data.AUTOTUNE)
    shuffled_ds = resized_ds.shuffle(
        buffer_size=total_examples,
        seed=seed,
        reshuffle_each_iteration=False,
    )

    train_raw = shuffled_ds.take(train_size)
    val_raw = shuffled_ds.skip(train_size).take(val_size)
    test_raw = shuffled_ds.skip(train_size + val_size)

    def cast_float32(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.cast(image, tf.float32), tf.cast(label, tf.int32)

    augmentation = get_augmentation_layer(seed=seed)

    def augment_batch(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return augmentation(images, training=True), labels

    train_ds = (
        train_raw.map(cast_float32, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(buffer_size=1000, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_raw.map(cast_float32, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_raw.map(cast_float32, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    split_counts = {
        "total": total_examples,
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }

    return train_ds, val_ds, test_ds, class_names, split_counts
