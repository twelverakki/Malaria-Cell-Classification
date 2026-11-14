# Malaria Cell Classification: Custom CNN vs MobileNetV2

Deep Learning course assignment comparing a **Custom CNN (Baseline)** and **Transfer Learning (`MobileNetV2`)** on the NIH/TFDS **Malaria** thin blood smear dataset (27,558 cell images, binary classification: `Parasitized` vs `Uninfected`).

## Repository Structure

```text
├── notebooks/
│   └── malaria_cnn_vs_mobilenetv2.ipynb   # Main Colab-ready notebook
├── src/
│   ├── __init__.py
│   ├── data.py                            # TFDS loading, deterministic splitting, augmentation
│   ├── models.py                          # Custom CNN & MobileNetV2 builders + callbacks
│   └── evaluate.py                        # Metrics, plots, comparison table
├── outputs/                               # Saved plots & evaluation summary
├── requirements.txt
├── SPEC.MD
└── README.md
```

## Methodology

- **Dataset:** `tensorflow_datasets` `malaria` (27,558 samples), split deterministically (`seed=42`) into **70% Train / 15% Validation / 15% Test**.
- **Preprocessing (`tf.data`):**
  - Resized to `(128, 128, 3)`.
  - Light augmentation on the **train split only**: random horizontal/vertical flip and `0.1` rotation.
  - `.cache()`, `.shuffle()`, `.batch(32)`, `.prefetch(tf.data.AUTOTUNE)`.
- **Model 1 — Custom CNN:** `Rescaling(1/255)` -> 3 blocks of `Conv2D` (`32 -> 64 -> 128`) + `BatchNormalization` + `MaxPooling2D` -> `GlobalAveragePooling2D` -> `Dropout(0.4)` -> `Dense(128, relu)` -> `Dense(1, sigmoid)`.
- **Model 2 — MobileNetV2:** ImageNet-pretrained `MobileNetV2` (`include_top=False`) with:
  - **Phase 1 (Feature Extraction):** Frozen base model, `GlobalAveragePooling2D` -> `Dropout(0.3)` -> `Dense(1, sigmoid)` for 10 epochs (`Adam`, `lr=1e-3`).
  - **Phase 2 (Fine-Tuning):** Top 30 layers unfrozen (with `BatchNormalization` kept frozen) for up to 10 epochs (`Adam`, `lr=1e-5`) with `EarlyStopping`.

## How to Run

### Option 1: Google Colab (Recommended — T4 GPU)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/twelverakki/Malaria-Cell-Classification/blob/main/notebooks/malaria_cnn_vs_mobilenetv2.ipynb)

1. Click **Open in Colab** above (or open `notebooks/malaria_cnn_vs_mobilenetv2.ipynb` in Colab).
2. Set runtime to **T4 GPU** (*Runtime -> Change runtime type -> T4 GPU*).
3. Run all cells — the setup cell automatically clones `https://github.com/twelverakki/Malaria-Cell-Classification.git` when running in Colab.
4. Generated plots and `model_comparison.csv` / `model_comparison.md` will be saved to `outputs/`.

### Option 2: Local Python 3.10+
```bash
pip install -r requirements.txt
jupyter notebook notebooks/malaria_cnn_vs_mobilenetv2.ipynb
```

## Results

Run the notebook to populate `outputs/model_comparison.md` and the visualizations in `outputs/`:
- `custom_cnn_history.png`, `custom_cnn_cm.png`, `custom_cnn_samples.png`
- `mobilenetv2_history.png`, `mobilenetv2_cm.png`, `mobilenetv2_samples.png`
- `model_comparison.csv` & `model_comparison.md`
