# 🧠 CNN for Race Classification (UTKFace Dataset)

This project builds a **Convolutional Neural Network (CNN)** using the **UTKFace dataset** to predict **race categories**: White, Black, Asian, Indian, Other.  
The model is trained in **Google Colab** due to CPU/RAM limitations.

---

👤 **Author:** Abdul Manan  
🧪 Plant Breeder | Machine & Deep Learning Researcher  
📧 [abdulmanan2287@gmail.com](mailto:abdulmanan2287@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/abdul-manan-0aa546332/) | 💻 [GitHub](https://github.com/manan348)

🗓️ **Last Updated:** February 2026

---

## 📂 Dataset Information

- **Source:** [UTKFace-Cropped Dataset – HuggingFace](https://huggingface.co/datasets/py97/UTKFace-Cropped?utm_source=chatgpt.com)  
- **Objective:** Predict **race** from facial images  
- **Number of images used:** 5,000 (subset due to RAM limits)  

---

## 🎯 Project Objective

Build a **deep learning CNN** for **multi-class classification (5 races)** using:

- **Hyperparameter tuning** (Keras Tuner)  
- **Batch Normalization**  
- **Dropout & L2 regularization**  
- **Class weighting** for imbalanced races  
- **Early stopping** to prevent overfitting  

---

## 🧠 CNN Modeling Pipeline

### 1️⃣ Data Preprocessing

- Selected **first 5,000 images**  
- Removed **corrupted images**, **incorrect filenames**, and **duplicates**  
- Resized images to `128x128` pixels  
- Normalized pixel values (0–1)  
- Assigned labels based on filenames (0–4 for races)  
- Split dataset into **training (80%)** and **testing (20%)**  

### 2️⃣ CNN Model Architecture

- 3 Conv2D layers + MaxPooling2D + BatchNormalization + Dropout  
- Dense layer + Dropout  
- Output layer: **Softmax** (5 classes)  
- L2 regularization on layers  
- Optimizer tuned via Keras Tuner (Adam or RMSprop)  
- Loss function: Sparse Categorical Crossentropy  
- Metric: Accuracy  

### 3️⃣ Hyperparameter Tuning

- **RandomSearch** from Keras Tuner  
- Tuned: Conv1/Conv2 filters, Dense units, Dropout rates, Optimizer  
- **Epochs:** 5 for tuning, 10 for final model  
- **Early stopping:** Patience=2 on validation loss  
- **Class weighting:** Applied to balance underrepresented races  

---

## 📊 Model Evaluation

- **Test Accuracy:** ~61.9% (5K images, CPU)  

### Confusion Matrix
- **White:** ~75–80%, some misclassifications  
- **Black & Asian:** ~80–90% accuracy  
- **Indian:** ~50–60% accuracy  
- **Other:** ~10–20%, low accuracy  

### ROC Curve (One-vs-Rest)
- **Black & Asian:** AUC ~0.90+  
- **White:** AUC ~0.83  
- **Indian:** AUC ~0.85  
- **Other:** AUC ~0.68  

### Precision-Recall Curve
- High precision: **Black, Asian, White**  
- Declining precision: **Indian**  
- Very low precision: **Other (~0.2)**  

**Insights:**  
- Model biased toward **White**  
- **Other** class underrepresented → needs more data  
- Fairness concerns exist; additional techniques recommended  

---

## ⚙️ Requirements

| Library        | Version (recommended)|
|----------------|----------------------|
| Python         | 3.8+                 |
| NumPy          | latest               |
| OpenCV         | latest               |
| Matplotlib     | latest               |
| Seaborn        | latest               |
| Scikit-learn   | latest               |
| TensorFlow     | 2.15+                |
| Keras          | latest               |
| Keras-tuner    | latest               |
| Google Colab   | recommended          |

Install via:

```bash
pip install -r requirements.txt
