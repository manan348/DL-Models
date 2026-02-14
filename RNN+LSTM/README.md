# 🧠 Recurrent Neural Network (RNN) for Twitter Airline Sentiment Analysis

This project implements a **Recurrent Neural Network (RNN)** model using **Long Short-Term Memory (LSTM)** to perform **sentiment classification** on airline-related tweets from Twitter.  
The goal is to classify each tweet as **Positive**, **Neutral**, or **Negative** based on its textual content.

---

📌 **Recommended Environment:**  
> Designed for **Google Colab (CPU/GPU)** users, but compatible with IDEs like **VS Code**, **PyCharm**, or **Jupyter Notebook**.

---

👤 **Author:** Abdul Manan  
🧪 *Plant Breeder | Machine & Deep Learning Researcher*  
📧 [abdulmanan2287@gmail.com](mailto:abdulmanan2287@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/abdul-manan-0aa546332/) | 💻 [GitHub](https://github.com/manan348)

🗓️ **Last Updated:** February 2026

---

## 📂 Dataset Information

- **Source:** [:contentReference[oaicite:0]{index=0}](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
- **Description:**  
  Tweets directed at six major U.S. airlines, labeled as *positive*, *neutral*, or *negative*.
- **Objective:**  
  Predict the **sentiment polarity** of tweets related to airline services.

---

## 🎯 Project Objective

> Develop a **deep learning NLP model** using a **Recurrent Neural Network (RNN)** with **Bidirectional LSTM** to accurately classify airline tweets into sentiment categories.  
> The model includes **data balancing**, **text preprocessing**, **emoji handling**, and **hyperparameter tuning** for optimal results.

---

## 🧩 Model Performance Summary

| Metric | Value |
|--------|--------|
| **Validation Accuracy** | **0.9141** |
| **Test Accuracy** | **0.9141** |
| **Macro F1-score** | **0.91** |
| **Weighted F1-score** | **0.91** |
| **Negative F1** | 0.89 |
| **Neutral F1** | 0.90 |
| **Positive F1** | 0.95 |

---

### 📈 Model Insights

- **Best Validation Accuracy:** 91.4%  
- **Stable generalization:** Training vs. Validation gap ≈ 4%  
- **High consistency:** Test accuracy matches validation accuracy  
- **Most accurate class:** Positive sentiment (F1 = 0.95)  
- **Most confused classes:** Negative ↔ Neutral (common in real tweets)

---

## 🔍 Exploratory Data Analysis (EDA)

**Visualizations created using `plotly.express`:**
- **Sentiment Distribution:** Shows majority of tweets are *negative* (~63%)  
- **Sentiment per Airline:** Reveals **United** and **US Airways** have most negative tweets  
- **Tweet Length Distribution:** Helps determine max sequence length for RNN tokenization  

📊 *Findings:*
- Tweets are short (median length ≈ 15 words)
- Class imbalance: Negative >> Neutral > Positive
- Imbalance addressed with **upsampling**

---

## 🧹 Text Preprocessing Pipeline

### 🧾 Cleaning Steps
1. Remove **URLs**, **hashtags**, **mentions (@user)**, and **punctuation**
2. Convert **emojis** to text equivalents (e.g., 😊 → "smiling face")
3. Expand **contractions** (e.g., "can’t" → "cannot")
4. Convert to lowercase and strip extra spaces

### ⚙️ Tools Used
- `emoji` – for emoji detection and conversion  
- `re` (regex) – for pattern-based cleaning  
- `contractions` – for expanding short forms  
- `LabelEncoder` – for encoding sentiment labels (0 = Negative, 1 = Neutral, 2 = Positive)

---

## ⚖️ Dataset Balancing

To mitigate class imbalance:
- Oversampled *Neutral* and *Positive* classes using **:contentReference[oaicite:1]{index=1}**
- Resulting dataset has **equal class distribution** across all sentiments.

---

## 🔢 Tokenization & Padding

| Parameter | Description | Value |
|------------|--------------|--------|
| **MAX_WORDS** | Vocabulary size | 12,000 |
| **MAX_LEN** | Max tweet length | 50 tokens |
| **Padding Type** | Post | Applied |
| **Tokenizer OOV Token** | `<OOV>` | Used for unseen words |

---

## 🧠 Model Architecture (RNN - LSTM)

| Layer | Description |
|--------|-------------|
| **Embedding** | Word embedding (trainable) |
| **Bidirectional LSTM** | Captures forward and backward sequence dependencies |
| **Dense (Softmax)** | 3 output classes (Positive, Neutral, Negative) |

**Optimizer:** Adam  
**Loss:** Sparse Categorical Crossentropy  
**Metrics:** Accuracy  

---

## ⚙️ Hyperparameter Tuning

Three models tested with different configurations:

| Model   | Embedding Dim  | LSTM Units  | Dropout  | Batch Size | Epochs |
|---------|----------------|-------------|----------|------------|--------|
| 1       | 32             | 32          | 0.2      | 32         | 5      |
| 2       | 64             | 64          | 0.3      | 64         | 7      |
| 3       | 128            | 128         | 0.3      | 128        | 7      |

**Best Model:** Model 3 → **Validation Accuracy = 91.4%**

### Callbacks Used
- `EarlyStopping(monitor='val_loss', patience=3)`  
- `ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)`

---

## 📊 Results & Visualization

### 🟦 Accuracy Curve
- Training accuracy increased steadily to **96%**
- Validation stabilized near **91%** (no overfitting)

### 🟧 Loss Curve
- Training loss dropped smoothly
- Validation loss plateaued → good regularization

### 🧮 Confusion Matrix
| Actual \ Predicted | Negative | Neutral | Positive |
|--------------------|-----------|----------|-----------|
| **Negative** | 1584 | 173 | 80 |
| **Neutral** | 105 | 1618 | 60 |
| **Positive** | 24 | 28 | 1798 |

✅ Correct predictions dominate diagonals  
⚠️ Minor confusion between *Negative* and *Neutral*

---

Install via:

```bash
pip install -r requirements.txt