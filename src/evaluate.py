"""Evaluation metrics, comparison tables, and visualization utilities."""

import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Compute test-set classification metrics and predictions for a binary model."""
    y_true_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_true_list.append(labels.numpy().ravel())
        y_prob_list.append(probs)

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob >= threshold).astype(int)

    # Note: In TFDS Malaria, 0 = Parasitized (positive disease class), 1 = Uninfected.
    # Standard binary metrics treat label 1 as positive; pos_label=0 reports sensitivity
    # for Parasitized detection if desired, while default macro/binary is also informative.
    metrics = {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
    }

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def build_comparison_table(
    results: List[Dict[str, object]],
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame and optionally save CSV & Markdown."""
    rows = [res["metrics"] for res in results]
    df = pd.DataFrame(rows).set_index("Model")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, "model_comparison.csv"))
        with open(os.path.join(save_dir, "model_comparison.md"), "w", encoding="utf-8") as f:
            f.write(df.round(4).to_markdown())

    return df


def merge_histories(
    hist1: tf.keras.callbacks.History,
    hist2: Optional[tf.keras.callbacks.History] = None,
) -> Tuple[Dict[str, List[float]], Optional[int]]:
    """Merge Phase 1 and optional Phase 2 Keras History objects."""
    merged: Dict[str, List[float]] = {k: list(v) for k, v in hist1.history.items()}
    split_epoch = None

    if hist2 is not None:
        split_epoch = len(merged["loss"])
        for key, values in hist2.history.items():
            if key in merged:
                merged[key].extend(values)
            else:
                merged[key] = list(values)

    return merged, split_epoch


def plot_training_history(
    history_dict: Dict[str, List[float]],
    model_name: str,
    save_path: Optional[str] = None,
    fine_tune_epoch: Optional[int] = None,
) -> None:
    """Plot Training vs Validation Accuracy and Loss curves."""
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy subplot
    axes[0].plot(epochs, history_dict["accuracy"], label="Train Accuracy", marker="o", markersize=4)
    axes[0].plot(epochs, history_dict["val_accuracy"], label="Val Accuracy", marker="s", markersize=4)
    if fine_tune_epoch is not None:
        axes[0].axvline(fine_tune_epoch + 0.5, color="gray", linestyle="--", label="Start Fine-Tuning")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss subplot
    axes[1].plot(epochs, history_dict["loss"], label="Train Loss", marker="o", markersize=4)
    axes[1].plot(epochs, history_dict["val_loss"], label="Val Loss", marker="s", markersize=4)
    if fine_tune_epoch is not None:
        axes[1].axvline(fine_tune_epoch + 0.5, color="gray", linestyle="--", label="Start Fine-Tuning")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Crossentropy Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix heatmap with counts and row-normalized percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
    annot = np.array(
        [[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    model_name: str,
    num_samples: int = 12,
    save_path: Optional[str] = None,
) -> None:
    """Visualize sample test images with true label, predicted label, and confidence score."""
    images_batch, labels_batch = next(iter(test_ds))
    images_np = images_batch[:num_samples].numpy()
    labels_np = labels_batch[:num_samples].numpy()

    probs = model.predict(images_np, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    cols = 4
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < len(images_np):
            img_display = np.clip(images_np[idx], 0, 255).astype(np.uint8)
            true_label = class_names[int(labels_np[idx])]
            pred_idx = int(preds[idx])
            pred_label = class_names[pred_idx]
            confidence = probs[idx] if pred_idx == 1 else (1.0 - probs[idx])
            is_correct = pred_idx == int(labels_np[idx])

            ax.imshow(img_display)
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
                color="green" if is_correct else "red",
                fontsize=10,
            )
        ax.axis("off")

    fig.suptitle(f"{model_name} — Sample Test Predictions", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
