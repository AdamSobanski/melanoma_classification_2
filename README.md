# Melanoma Classification

Dermoscopic image classification (melanoma vs. benign lesion) using transfer learning on EfficientNet.

## Project Goal

Detecting melanoma in dermoscopic images, with a particular focus on **high recall** for the malignant class — in a medical context, a missed malignant case is a far more costly error than a false alarm.

## Results

| Model | Recall (malignant) | Precision | AUC |
|---|---|---|---|
| EfficientNet-B0 (baseline) | 84.4% | — | — |
| **EfficientNet-B4 (current)** | **93.9%** | 89.6% | 0.980 |

Tested on an independent sample of 505 images with confirmed malignant diagnosis.

The ready-to-use model is hosted on Hugging Face:
**[Ai-Adam-Six-Sigma/melanoma-efficientnet-b4](https://huggingface.co/Ai-Adam-Six-Sigma/melanoma-efficientnet-b4)**

## Repository Structure

```
notebooks/
├── eda-melanomoa-isic-2019-2020.ipynb       # data exploration (EDA)
├── melanoma_training_b4.ipynb                # training notebook (draft version)
├── trening-efficientnet-b4-isic-2019-2020-mel-vs.ipynb  # final training notebook (run on Kaggle)
└── compare_b0_b4.ipynb                       # B0 vs B4 model comparison
```

Model weights (`.pt`) and training data **are not part of this repository** — models are hosted on Hugging Face, and the data comes from the public ISIC dataset.

## Data

- **Source:** [ISIC 2019 + 2020](https://www.kaggle.com/datasets/qikangdeng/isic-2019-and-2020-melanoma-dataset) (Kaggle, mel vs nevus)
- ~11,400 images after deduplication (48 duplicates removed via MD5 hash)
- Class distribution: mel 44.6% / nevus 55.4%

## Methodology

1. **EDA** — checked class distribution, duplicates, and image sizes. Found that the nevus class had on average ~3x higher resolution than mel in the source data — a potential risk of the model learning that difference instead of actual lesion features.
2. **Class weighting** — correcting the mild class imbalance in the loss function (`BCEWithLogitsLoss` with `pos_weight`).
3. **Corrective augmentation** — random image quality degradation (simulating lower resolution) to mitigate the risk from point 1.
4. **Training** — EfficientNet-B4 (`timm`), ImageNet transfer learning, 380x380 input, checkpointing + early stopping (patience=5 epochs). Training ran for 13 epochs before early stopping triggered; the best checkpoint (highest recall) was saved at epoch 8.
5. **Evaluation** — on an ISIC hold-out set and on an independent sample of 505 real malignant images, for comparison against the earlier B0 model.

## Limitations

- The model is a decision-support tool, not a substitute for medical diagnosis.
- ~6% of true malignant cases are still missed (93.9% recall, not 100%).
- No formal testing has been done for demographic bias (skin type, age, lesion location).

## Author

Adam Sobański
