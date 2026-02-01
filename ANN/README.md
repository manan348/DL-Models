# 🧠 Artificial Neural Network (ANN) for Bank Customer Service Prediction

This project applies an **Artificial Neural Network (ANN)** model to predict customer responses using **Bank Marketing data**.  
The study focuses on handling **class imbalance** using threshold tuning and evaluating model performance with various metrics.

---

** This is not IDE users but for google-collab users recommended.**


👤 **Author:** Abdul Manan  
🧪 *Plant Breeder | Machine & Deep Learning Researcher*  
📧 [abdulmanan2287@gmail.com](mailto:abdulmanan2287@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/abdul-manan-0aa546332/) | 💻 [GitHub](https://github.com/manan348)

🗓️ **Last Updated:** February 2026

---

## 📂 Dataset Information

- **Source:** [Bank Marketing Dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **DOI:** [https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
- **Objective:** Predict whether a bank customer will subscribe to a term deposit based on campaign data.

---

## 🎯 Project Objective

> Develop a deep learning model using an **ANN** to predict customer behavior.  
> Multiple probability **thresholds** were tested to mitigate the effect of **class imbalance** and improve recall for the minority class.

---

## 🧩 ANN Model Performance by Threshold

| Threshold | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Accuracy | Weighted Avg F1 | Macro Avg F1 |
|------------|---------------------|------------------|--------------------|-----------|-----------------|---------------|
| 0.30       | 0.605              | 0.382           | 0.468             | 0.898     | 0.888           | 0.706         |
| 0.34       | 0.706              | 0.229           | 0.345             | 0.899     | 0.875           | 0.645         |
| 0.50       | 0.821              | 0.052           | 0.098             | 0.888     | 0.842           | 0.519         |

---

### 📈 Observations

- As the **threshold increases**, precision improves while recall drops sharply.  
- The **best trade-off** occurs near a threshold of **0.3**, where the model maintains a balance between accuracy (0.898) and recall (0.382).  
- Lower thresholds capture more positive (minority) cases — beneficial in imbalanced banking data scenarios.

---

## 🔍 Exploratory Data Analysis (EDA)

- Dataset shows **no major missing data or anomalies**.
- Several features have **outliers causing skewness**.
- Categorical features such as `housing`, `loan`, and `month` were **encoded** for model input.
- **Feature scaling** applied to numerical columns for stable ANN training.

---

## 🧠 Modeling Pipeline

### 1️⃣ Data Preprocessing
- Outlier inspection and normalization  
- Encoding of categorical features  
- Feature scaling for numerical columns  

### 2️⃣ Handling Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance training data.

### 3️⃣ ANN Model Building
- **Frameworks:** TensorFlow / Keras  
- **Optimizer:** Adam  
- **Loss Function:** Binary Cross-Entropy  
- **Metrics:** Accuracy, AUC  
- **Tuning:** Hyperparameters optimized with **Keras Tuner (RandomSearch)**

### 4️⃣ Evaluation
- Compared results across thresholds (0.3–0.5)
- Evaluated with Precision, Recall, F1, AUC, and Accuracy

---

## 📊 Visualizations

- **Correlation Heatmap** – for feature relationships  
- **ROC Curve** – model discrimination ability  
- **Precision–Recall Curve** – imbalance analysis  

> All visualizations are color-blind-friendly and publication-ready.  
> Plots are saved in the `/outputs` directory.

---

## ⚙️ Requirements

| Library | Version (recommended) |
|----------|----------------------|
| Python | 3.8+ |
| NumPy | latest |
| Pandas | latest |
| Matplotlib | latest |
| Seaborn | latest |
| Scikit-learn | latest |
| TensorFlow | 2.15+ |
| Keras | latest |
| Imbalanced-learn | latest |
| Keras-tuner | latest |
| Jupyter Notebook | optional |

Install all dependencies with:
```bash
pip install -r requirements.txt
