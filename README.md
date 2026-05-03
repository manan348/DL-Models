# DL-Models

**Author:** Abdul Manan — Plant Breeder | Machine & Deep Learning Researcher  
**Contact:** abdulmanan2287@gmail.com · [LinkedIn](https://linkedin.com) · [GitHub](https://github.com/manan348)  
**Last Updated:** February 2026  
**License:** MIT

---

A hands-on collection of deep learning projects covering the three core neural network architectures — **ANN**, **CNN**, and **RNN+LSTM** — built for learners and researchers who want working, well-documented implementations to study or build on.

---

## Repository Structure

```
DL-Models/
├── ANN/                   ← Bank customer subscription prediction
├── CNN/
│   ├── CNN_Binary/        ← Gender classification from facial images
│   ├── CNN_MultiClass/    ← Race classification from facial images
│   └── CNN_Regression/    ← Age prediction from facial images
├── RNN+LSTM/              ← Twitter airline sentiment analysis
└── README.md
```

Each project folder follows a consistent layout:

```
Project/
├── Dataset/        ← Raw data (or download instructions)
├── Notebook/       ← Jupyter notebook with full pipeline
├── outputs/        ← Plots, curves, confusion matrices
├── requirements.txt
└── README.md
```

---

## Projects

### 1. ANN — Bank Customer Subscription Prediction

**Task:** Binary classification — predict whether a customer subscribes to a term deposit.  
**Dataset:** [UCI Bank Marketing Dataset](https://doi.org/10.24432/C5K306) (CC BY 4.0)

**Pipeline highlights:**
- EDA with outlier handling, categorical encoding, and feature scaling
- Class imbalance handled via **SMOTE**
- TensorFlow/Keras ANN with Adam optimizer and Binary Cross-Entropy loss
- Hyperparameter tuning via **Keras Tuner (RandomSearch)**
- Threshold analysis to optimize precision-recall trade-off

**Results:**

| Threshold | Precision (Class 1) | Recall (Class 1) | F1 | Accuracy |
|-----------|---------------------|------------------|----|----------|
| 0.30      | 0.605               | 0.382            | 0.468 | 89.8% |
| 0.34      | 0.706               | 0.229            | 0.345 | 89.9% |
| 0.50      | 0.821               | 0.052            | 0.098 | 88.8% |

> Best trade-off at threshold **0.30** — balances recall and accuracy for this imbalanced dataset.

**Outputs:** Correlation heatmap · ROC curve · Precision-Recall curve

---

### 2. CNN — Facial Image Analysis (UTKFace Dataset)

All three sub-projects use the [UTKFace-Cropped Dataset](https://susanqq.github.io/UTKFace/) and share the same architecture skeleton (Conv2D blocks → BatchNorm → Dropout → Dense output).

---

#### CNN_Binary — Gender Prediction

**Task:** Binary classification (Male / Female)  
**Data:** 5,000 images (128×128), sampled from 20k+

**Architecture:** 2× Conv2D + MaxPooling + Dropout → Dense → Sigmoid  
**Regularization:** L2 + Dropout · **Optimizer:** Adam/RMSprop (Keras Tuner)

**Results:**
- Training accuracy: ~94–96% · Test accuracy: ~79–80%
- AUC = **0.89**
- Male detection: 70–76% · Female detection: 82–89%

**Saved model:** `best_gender_model.h5`  
**Outputs:** Accuracy/Loss curves · Confusion matrix · ROC curve · Precision-Recall curve

---

#### CNN_MultiClass — Race Classification

**Task:** 5-class classification (White, Black, Asian, Indian, Other)  
**Data:** 5,000 images

**Architecture:** 3× Conv2D + MaxPooling + BatchNorm + Dropout → Dense → Softmax  
**Loss:** Sparse Categorical Cross-Entropy · Class weighting applied

**Results:**
- Test accuracy: **~61.9%**
- Best AUC: Black & Asian (~0.90+) · Worst: "Other" (~0.68)
- Note: "Other" class is heavily underrepresented; model shows bias toward White class

**Outputs:** Confusion matrix · ROC curve (One-vs-Rest) · Precision-Recall curve

---

#### CNN_Regression — Age Prediction

**Task:** Continuous age regression  
**Data:** 7,000 images (largest subset — regression needs more data)

**Architecture:** 3× Conv2D + MaxPooling + BatchNorm + Dropout → Dense → Linear  
**Loss:** MSE · **Metric:** MAE · Ages normalized to [0, 1]

**Data augmentation:** Rotation ±15° · Width/height shift ±10% · Horizontal flip

**Results:**
- Strong predictions for ages **0–40**
- Ages >70 are underpredicted due to data imbalance

**Outputs:** MSE/MAE loss curves · Actual vs. Predicted scatter plot

---

### 3. RNN+LSTM — Twitter Airline Sentiment Analysis

**Task:** Multi-class sentiment classification — Positive / Neutral / Negative  
**Dataset:** [Kaggle Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (~63% negative)

**Pipeline highlights:**
- EDA with Plotly — sentiment distribution, per-airline breakdown, tweet length analysis
- Text preprocessing: URL/hashtag/mention removal, emoji-to-text conversion, contraction expansion, lowercasing
- Class balancing: upsampled Neutral & Positive to match Negative
- Tokenization: vocab=12,000, max_len=50, OOV token for unseen words
- **Architecture:** Embedding → Bidirectional LSTM → Dense (Softmax)
- **Callbacks:** EarlyStopping (patience=3) · ReduceLROnPlateau (factor=0.3, patience=2)

**Model configurations tested:**

| Model | Embedding Dim | LSTM Units | Dropout | Batch Size | Epochs |
|-------|---------------|------------|---------|------------|--------|
| 1 | 32 | 32 | 0.2 | 32 | 5 |
| 2 | 64 | 64 | 0.3 | 64 | 7 |
| **3 ★** | **128** | **128** | **0.3** | **128** | **7** |

**Best model (Model 3) results:**

| Metric | Score |
|--------|-------|
| Validation Accuracy | 91.4% |
| Test Accuracy | 91.4% |
| Macro F1 | 0.91 |
| Negative F1 | 0.89 |
| Neutral F1 | 0.90 |
| Positive F1 | 0.95 |

**Outputs:** Training accuracy/loss curves · Confusion matrix · Tweet length distribution · Sentiment distribution · Per-airline sentiment · Word frequency distribution (log scale)

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.8+ | Core language |
| TensorFlow 2.15+ / Keras | Model building & training |
| Keras Tuner | Hyperparameter optimization (RandomSearch) |
| Scikit-learn | Preprocessing, metrics (ROC, F1, confusion matrix) |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Static visualizations |
| Plotly Express | Interactive EDA (RNN project) |
| Imbalanced-learn | SMOTE for class balancing (ANN) |
| OpenCV | Image loading and resizing (CNN) |
| emoji / contractions | NLP text preprocessing (RNN) |
| Google Colab | Recommended training environment |

---

## Project Summary

| Project | Type | Task | Dataset | Best Metric |
|---------|------|------|---------|-------------|
| ANN | Binary Classification | Bank subscription | UCI Bank Marketing | 89.9% acc |
| CNN_Binary | Binary Classification | Gender from face | UTKFace (5K) | ~80% acc · AUC 0.89 |
| CNN_MultiClass | Multi-Class | Race from face | UTKFace (5K) | ~61.9% acc |
| CNN_Regression | Regression | Age from face | UTKFace (7K) | MAE (normalized) |
| RNN+LSTM | Multi-Class | Twitter sentiment | Kaggle Airline Tweets | 91.4% acc · F1 0.91 |

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/manan348/DL-Models.git
cd DL-Models

# Install dependencies for a specific project
pip install -r ANN/requirements.txt

# Open the notebook
jupyter notebook ANN/Notebook/ANN.ipynb
```

> Recommended: Run notebooks on **Google Colab** for free GPU access.

---

## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  
The Bank Marketing Dataset is licensed under **CC BY 4.0** (UCI ML Repository).
