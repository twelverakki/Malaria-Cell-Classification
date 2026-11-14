"""Model builders and training callbacks for Custom CNN and MobileNetV2."""

from typing import List, Sequence, Tuple
import tensorflow as tf


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    conv_filters: Sequence[int] = (32, 64, 128),
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile the baseline Custom CNN for binary malaria classification."""
    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    for idx, filters in enumerate(conv_filters, start=1):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name=f"conv_{idx}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{idx}")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"pool_{idx}")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Custom_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_mobilenetv2(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Build and compile MobileNetV2 with a frozen ImageNet backbone (Phase 1)."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    # Equivalent to tf.keras.applications.mobilenet_v2.preprocess_input for [0, 255] inputs
    x = tf.keras.layers.Rescaling(
        1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess"
    )(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base_model


def unfreeze_for_finetuning(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    unfreeze_top_n: int = 30,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Unfreeze the top N layers of MobileNetV2 (keeping BatchNorm frozen) for Phase 2."""
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - unfreeze_top_n)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_callbacks(
    patience: int = 3,
    monitor: str = "val_loss",
) -> List[tf.keras.callbacks.Callback]:
    """Create EarlyStopping and ReduceLROnPlateau callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
