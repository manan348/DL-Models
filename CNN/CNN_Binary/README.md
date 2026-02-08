# 🧠 Convolutional Neural Network (CNN) for Gender Prediction

This project builds a **CNN model** using the **UTKFace dataset** to predict **gender (Male/Female)**.  
The model is trained in **Google Colab** due to GPU requirements and RAM constraints.

---

👤 **Author:** Abdul Manan  
🧪 *Plant Breeder | Machine & Deep Learning Researcher*  
📧 [abdulmanan2287@gmail.com](mailto:abdulmanan2287@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/abdul-manan-0aa546332/) | 💻 [GitHub](https://github.com/manan348)

🗓️ **Last Updated:** February 2026

## 📂 Dataset Information

- **Source:** [UTKFace-Cropped Dataset – HuggingFace](https://huggingface.co/datasets/py97/UTKFace-Cropped?utm_source=chatgpt.com)
- **License:** Check dataset source
- **Objective:** Predict **gender** from facial images.
- **Number of images used:** 5,000 for training/testing (original dataset >20k images)

## 🎯 Project Objective

> Build a deep learning CNN model for **binary classification (Male/Female)**.  
> Use **hyperparameter tuning (Keras Tuner)**, **Dropout**, **L2 regularization**, and **early stopping** to improve performance and reduce overfitting.

## 🧠 CNN Modeling Pipeline

### 1️⃣ Data Preprocessing
- Load images and resize to `128x128` pixels  
- Normalize pixel values (0-1)  
- Assign labels based on filenames (0=Male, 1=Female)  
- Split into training (80%) and testing (20%)  

### 2️⃣ Handling Dataset Issues
- Removed **corrupted images**, **incorrect filenames**, and **duplicate images**  
- Selected first 5K images for training due to RAM limits  

### 3️⃣ CNN Model Building
- **Frameworks:** TensorFlow / Keras  
- **Layers:**
  - Conv2D + MaxPooling2D + Dropout (2 convolutional layers)
  - Flatten
  - Dense + Dropout
  - Output layer with Sigmoid activation  
- **Regularization:** L2 and Dropout
- **Optimizer:** Tuned via Keras Tuner (Adam or RMSprop)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

### 4️⃣ Hyperparameter Tuning
- **Keras Tuner:** RandomSearch  
- **Tunable parameters:** Conv1/Conv2 filters, dense units, dropout rates, optimizer
- **Epochs:** 5 (tuner search) + 10 (best model training)
- **Early stopping:** Stop training when validation loss stops improving

## 📊 Model Evaluation

- **Training Accuracy:** ~94-96%  
- **Validation Accuracy:** ~78-80%  
- **Test Accuracy:** ~79-80%  

### Confusion Matrix
- Males detected: 70-76%  
- Females detected: 82-89%  
- Slight class imbalance favoring female predictions

### ROC Curve
- **AUC = 0.89** (high discriminative ability)

### Precision-Recall Curve
- Shows trade-off between recall and precision for gender predictions

### Visualizations
- Training Accuracy vs Epoch
- Training Loss vs Epoch
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

> Plots are saved in `/outputs` folder for reference.

## ⚙️ Requirements

| Library | Version (recommended) |
|----------|----------------------|
| Python | 3.8+ |
| NumPy | latest |
| OpenCV | latest |
| Matplotlib | latest |
| Seaborn | latest |
| Scikit-learn | latest |
| TensorFlow | 2.15+ |
| Keras | latest |
| Keras-tuner | latest |
| Google Colab | recommended |

Install dependencies via:
```bash
pip install -r requirements.txt


---

## **7️⃣ Saving the Model**

```markdown
- Model saved as `best_gender_model.h5` in Google Drive for inference.

