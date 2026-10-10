# British Airways Review Sentiment Analysis

Binary sentiment classifier for British Airways customer reviews 
using VADER for labeling and classical ML models for classification.

## 📊 Dataset
- **Source:** BA_reviews.csv
- **Size:** 3,000 reviews
- **Columns:** `reviews` (raw text)

## 🔧 Pipeline

1. **Text cleaning** — lowercase, remove tags/punctuation/stopwords
2. **Sentiment labeling** — VADER compound score → positive / negative / neutral
3. **EDA** — visualized class distribution
4. **Drop neutral** — only 1.8% of data, too small to learn
5. **Modeling** — 4 classifiers trained on TF-IDF features
6. **Deployment** — best pipeline saved as `sentiment_model.pkl`

## 🤖 Model Comparison

| Model                | Accuracy |
|----------------------|----------|
| LogisticRegression   | 78%      |
| MultinomialNB        | 71%      |
| SVC (RBF)            | 80%      |
| **LinearSVC**        | **82%** ✅ |

**Best model:** LinearSVC at **82% accuracy** on the held-out test set.

## 🚀 Usage

```python
import joblib

pipe = joblib.load("sentiment_model.pkl")
pipe.predict(["The flight was delayed and baggage lost"])
# → array(['negative'], dtype=object)

pipe.predict(["Amazing crew, love it"])
# → array(['positive'], dtype=object)
