# British Airways Review Sentiment Analysis ✈️

## Project Overview

This project focuses on analyzing customer reviews of British Airways using Natural Language Processing (NLP) and machine learning techniques. The goal is to understand customer sentiment by classifying reviews as **positive or negative** and exploring the opinions customers express about their travel experiences.

The project includes data exploration, text preprocessing, sentiment labeling, machine learning model comparison, model evaluation, and the development of a reusable prediction pipeline.

## Objectives

- Analyze British Airways customer reviews to understand overall sentiment.
- Clean and preprocess textual data for machine learning.
- Use VADER sentiment analysis to generate sentiment labels.
- Convert text into numerical features using TF-IDF vectorization.
- Train and evaluate multiple machine learning classifiers.
- Build a pipeline for predicting sentiment from raw customer reviews.
- Save the trained pipeline for future use.

## Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — machine learning, TF-IDF, pipelines, and evaluation
- **VADER Sentiment** — rule-based sentiment analysis
- **Joblib** — saving the trained model
- **Jupyter Notebook** — interactive development and experimentation

## Project Workflow

### 1. Data Loading and Exploration

The project loads the `BA_reviews.csv` dataset and explores its structure, columns, and contents using Pandas. Exploratory data analysis (EDA) is used to understand the review data and visualize the distribution of sentiment labels.

### 2. Text Preprocessing

Customer reviews are cleaned before analysis. The preprocessing steps include:

- Converting text to lowercase.
- Removing verification-related phrases such as `Trip verified` and `Not verified`.
- Removing punctuation, numbers, and non-alphabetic characters.
- Removing extra whitespace.
- Removing English stop words.
- Filtering out words with fewer than three characters.

The cleaned reviews are stored in a separate column for further analysis.

### 3. Sentiment Labeling with VADER

VADER is used to calculate a compound sentiment score for each cleaned review. The scores are converted into sentiment categories using the following thresholds:

| Compound Score | Sentiment |
|---|---|
| 0.05 or greater | Positive |
| -0.05 or less | Negative |
| Between -0.05 and 0.05 | Neutral |

The neutral class is subsequently removed because it represents a small proportion of the dataset in the notebook. The machine learning models therefore focus on binary classification: positive versus negative sentiment.

**Note:** These labels are generated using VADER rather than manually annotated ground-truth labels.

### 4. Feature Extraction with TF-IDF

The cleaned text is transformed into numerical features using `TfidfVectorizer` from Scikit-learn.

The vectorizer is configured with a maximum of 5,000 features, allowing the models to learn from the importance of words in the review dataset.

### 5. Machine Learning Models

The notebook explores and compares the following classifiers:

- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Support Vector Classifier (SVC)**
- **Linear Support Vector Classifier (LinearSVC)**

The dataset is split into 80% training data and 20% testing data, using a random state of 42.

### 6. Model Evaluation

The trained classifiers are evaluated using classification reports and confusion matrices.

The evaluation considers metrics such as:

- Precision
- Recall
- F1-score
- Classification performance across positive and negative reviews

The notebook also visualizes confusion matrices to help identify correct and incorrect predictions.

### 7. Prediction Pipeline and Model Saving

A Scikit-learn pipeline is created to combine text cleaning, TF-IDF vectorization, and Logistic Regression into one reusable workflow.

The final pipeline uses class balancing and an increased iteration limit to help address class imbalance and support model convergence.

The trained pipeline is saved as:

`sentiment_model.pkl`

This allows the model to be loaded and used for future predictions without manually repeating each preprocessing step.

## Example Predictions

The pipeline can classify new customer reviews directly from raw text.

**Example 1: Negative review**

```python
pipe.predict([
    "The flight was delayed and baggage lost"
])
```

**Example 2: Positive review**

```python
pipe.predict([
    "Amazing crew, love it"
])
```

The pipeline returns a predicted sentiment label for each review.

## Installation and Setup

### Prerequisites

Make sure Python and Jupyter Notebook are installed on your computer.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn vaderSentiment joblib jupyter
```

### 3. Add the dataset

Place the `BA_reviews.csv` dataset in the same directory as the notebook.

The notebook expects the dataset to be available at the following path:

```text
BA_reviews.csv
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `sentiment_model.ipynb` and run the cells in order.

## Project Structure

```text
British-Airways-Sentiment-Analysis/
│
├── sentiment_model.ipynb   # Data analysis, training, and evaluation
├── BA_reviews.csv          # Input dataset (add separately)
├── sentiment_model.pkl     # Saved model pipeline (generated)
└── README.md               # Project documentation
```

The CSV dataset and saved model should be included in the repository only when appropriate for their size, licensing, and distribution permissions.

## Key Learnings

This project demonstrates practical experience with:

- Text preprocessing and feature engineering.
- Rule-based sentiment analysis using VADER.
- TF-IDF text vectorization.
- Training and comparing machine learning classifiers.
- Evaluating classification models.
- Building reusable Scikit-learn pipelines.
- Saving trained machine learning models with Joblib.
- Applying NLP techniques to customer feedback analysis.

## Limitations and Future Improvements

Potential improvements include:

- Use stratified train-test splitting and cross-validation.
- Compare models using a consistent evaluation dataset and report their actual scores.
- Tune model hyperparameters to improve performance.
- Experiment with advanced NLP approaches such as transformer-based sentiment classifiers.
- 
## Conclusion

This project demonstrates how NLP and machine learning can be applied to airline customer feedback to identify positive and negative sentiment. By combining text preprocessing, VADER-based labeling, TF-IDF feature extraction, model comparison, and a reusable prediction pipeline, the project provides a foundation for further customer review analysis.

## Author

**Onah Emmanuel Obinna**

- GitHub: [Your GitHub Profile](https://github.com/YOUR-USERNAME)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/)

