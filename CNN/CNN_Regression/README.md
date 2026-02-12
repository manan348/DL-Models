# 🧠 CNN for Age Prediction (UTKFace Dataset)

This project builds a **Convolutional Neural Network (CNN)** using the **UTKFace dataset** to predict **age** (regression problem).  
The model is trained in **Google Colab** due to CPU/RAM limitations.

---

👤 **Author:** Abdul Manan  
🧪 Plant Breeder | Machine & Deep Learning Researcher  
📧 [abdulmanan2287@gmail.com](mailto:abdulmanan2287@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/abdul-manan-0aa546332/) | 💻 [GitHub](https://github.com/manan348)

🗓️ **Last Updated:** February 2026

---

## 📂 Dataset Information

- **Source:** [UTKFace-Cropped Dataset – HuggingFace](https://huggingface.co/datasets/py97/UTKFace-Cropped?utm_source=chatgpt.com)  
- **Objective:** Predict **age** from facial images  
- **Number of images used:** 7,000 (subset due to RAM limits)  

---

## 🎯 Project Objective

Build a **deep learning CNN** for **age regression** using:

- **Hyperparameter tuning** (Keras Tuner)  
- **Batch Normalization**  
- **Dropout & L2 regularization**  
- **Data augmentation** to improve generalization  
- **Early stopping** to prevent overfitting  

---

## 🧠 CNN Modeling Pipeline

### 1️⃣ Data Preprocessing

- Selected **first 7,000 images**  
- Removed **corrupted images**, **incorrect filenames**, and **duplicates**  
- Resized images to `128x128` pixels  
- Normalized pixel values (0–1)  
- Assigned labels based on filenames (age)  
- Scaled ages to **0–1** (normalized by max age)  
- Split dataset into **training (80%)** and **testing (20%)**  

### 2️⃣ CNN Model Architecture

- 3 Conv2D layers + MaxPooling2D + BatchNormalization + Dropout  
- Flatten → Dense layer + Dropout  
- Output layer: **Linear activation** (regression)  
- L2 regularization on layers  
- Optimizer tuned via Keras Tuner (Adam or RMSprop)  
- Loss function: Mean Squared Error (MSE)  
- Metric: Mean Absolute Error (MAE)  

### 3️⃣ Hyperparameter Tuning

- **RandomSearch** from Keras Tuner  
- Tuned: Conv1/Conv2 filters, Dense units, Dropout rates, Optimizer  
- **Epochs:** 5 for tuning, 10 for final model  
- **Early stopping:** Patience=2 on validation loss  

### 4️⃣ Data Augmentation

- Rotation: ±15°  
- Width & height shift: ±10%  
- Horizontal flip  
- Helps generalization and reduces overfitting on small dataset  

---

## 📊 Model Evaluation

### Training vs Validation Loss (MSE)  
Shows how the model's **Mean Squared Error** changed during training and validation. Early epochs may show some instability, but the model generally converges by the end of training.

### Training vs Validation MAE  
Shows the **Mean Absolute Error** during training and validation. MAE gives a more interpretable measure of how far predictions are from actual ages (in normalized scale or years after rescaling).

### Actual vs Predicted Age  
Scatter plot of the model’s predictions vs true ages.  
- Predictions cluster well for ages 0–40  
- Older ages (>70) are underpredicted due to **data imbalance**  
- Highlights model bias toward younger age ranges

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
