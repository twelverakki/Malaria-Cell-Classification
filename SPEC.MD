# Project Specification: Malaria Cell Classification (CNN vs Transfer Learning)

## 1. Objective

Build a clean, reproducible Deep Learning project comparing **2 architectures** on the **Malaria** cell image dataset to satisfy the Deep Learning course assignment (minimum 2 architectures, hosted on GitHub).

- **Model 1 (Baseline):** Custom Convolutional Neural Network (CNN)
- **Model 2 (Transfer Learning):** Pretrained `MobileNetV2` (ImageNet weights) with frozen base + fine-tuning head
- **Target Environment:** Google Colab (T4 GPU) or local Python 3.10+ environment

## 2. Dataset & Preprocessing

- **Source:** [TensorFlow Datasets - Malaria](https://www.tensorflow.org/datasets/catalog/malaria) ([Original NIH Publication](https://lhncbc.nlm.nih.gov/publication/pub9932))
- **Total Samples:** 27,558 segmented thin blood smear cell images (balanced binary classes: `0 = Parasitized`, `1 = Uninfected` — verify label mapping via `ds_info.features['label'].names`).
- **Data Splitting:** `tfds` provides only a `train` split by default. Split deterministically (`seed=42`) into:
    - `70%` Train
    - `15%` Validation
    - `15%` Test
- **Preprocessing Pipeline (`tf.data`):**
    - Resize all images to `(128, 128, 3)` (raw images have varying dimensions).
    - Normalize pixel values to `[0, 1]` for Custom CNN, and use `tf.keras.applications.mobilenet_v2.preprocess_input` (or `Rescaling` layer inside the model) for `MobileNetV2`.
    - Apply light data augmentation on the **train split only**: random horizontal/vertical flip and small rotation (`0.1`).
    - Use `.cache()`, `.shuffle()`, `.batch(32)`, and `.prefetch(tf.data.AUTOTUNE)`.

## 3. Model Architectures

### A. Custom CNN (Baseline)

- 3–4 `Conv2D` + `BatchNormalization` + `MaxPooling2D` blocks (filters: `32 -> 64 -> 128`)
- `GlobalAveragePooling2D` (or `Flatten`) + `Dropout(0.3–0.5)` + `Dense(128, activation='relu')`
- Output: `Dense(1, activation='sigmoid')`

### B. Transfer Learning (`MobileNetV2`)

- Base model: `tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')`
- **Phase 1 (Feature Extraction):** Freeze base model (`trainable=False`), train classification head (`GlobalAveragePooling2D` -> `Dropout(0.3)` -> `Dense(1, activation='sigmoid')`) for 10 epochs.
- **Phase 2 (Optional Fine-Tuning):** Unfreeze top ~30 layers of base model, compile with low learning rate (`1e-5`), train for 5–10 epochs with `EarlyStopping`.

## 4. Training & Evaluation Requirements

- **Loss & Optimizer:** `BinaryCrossentropy` + `Adam` (`lr=1e-3` initial).
- **Callbacks:** `EarlyStopping(patience=3, restore_best_weights=True)` and `ReduceLROnPlateau`.
- **Required Metrics (on Test Set):**
    - Accuracy, Precision, Recall (Sensitivity), F1-Score, ROC-AUC
    - Side-by-side comparison table between Custom CNN and MobileNetV2
- **Required Visualizations (saved to `outputs/`):**
    - Training vs Validation Accuracy & Loss curves for both models
    - Confusion Matrix for both models
    - Sample test predictions (showing true label, predicted label, and confidence score)

## 5. Repository Structure

```text
malaria-deep-learning/
├── notebooks/
│   └── malaria_cnn_vs_mobilenetv2.ipynb   # Main Colab-ready notebook
├── src/
│   ├── data.py                            # TFDS loading, splitting, augmentation
│   ├── models.py                          # Custom CNN & MobileNetV2 builders
│   └── evaluate.py                        # Metrics, plots, comparison table
├── outputs/                               # Saved plots & evaluation summary
├── requirements.txt
└── README.md                              # Methodology, results table, how to run
```
