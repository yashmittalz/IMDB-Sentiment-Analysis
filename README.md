
# IMDB Sentiment Analysis Using NLP (Natural Language Processing)

## Overview

This project is a Natural Language Processing (NLP) initiative that performs sentiment analysis on IMDB movie reviews. It aims to classify reviews as either positive or negative using machine learning models, with a focus on preprocessing raw text data and converting it into numerical formats for predictive analysis.

## Features

- Cleans and preprocesses raw text data by removing HTML tags, punctuations, and stopwords.
- Converts text into numerical data using a Bag of Words model.
- Implements a Random Forest classifier for sentiment classification.
- Measures performance using the ROC AUC score.
- Provides an end-to-end pipeline for data preprocessing, model training, and evaluation.

## Project Goals

- Build an NLP pipeline for sentiment analysis.
- Create a robust model to classify movie reviews.
- Demonstrate practical applications of NLP techniques in real-world scenarios.
- Showcase the effectiveness of the Bag of Words model and Random Forest classifier.

## Defining NLP

Natural Language Processing (NLP) enables computers to understand, interpret, and respond to human language. In this project, NLP techniques are used to preprocess text data and extract meaningful features for sentiment analysis.

## Application of NLP

This project applies NLP in the following ways:

1. HTML tag removal to clean text data.
2. Text normalization by converting to lowercase and removing punctuations.
3. Stopword removal to focus on meaningful words.
4. Feature extraction using the Bag of Words model.
5. Sentiment classification with machine learning.


## Predictive Examples

Input: "This movie was fantastic! The plot and characters were well-developed."  
Output: Positive Sentiment.

Input: "The movie was too slow and boring for my taste."  
Output: Negative Sentiment.

## Requirements

- Python 3.x
- Libraries: numpy, pandas, matplotlib, sklearn, bs4, nltk, re
- NLTK stopwords dataset

## Usage

1. Import the required libraries.
2. Download the dataset `labeledTrainData.tsv` from Kaggle.
3. Preprocess the data by cleaning HTML tags, removing punctuation, and stopwords.
4. Train a Random Forest model using the Bag-of-Words features.
5. Evaluate model accuracy using the ROC AUC score.

## Dataset

- Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/ymanojkumar023/kumarmanoj-bag-of-words-meets-bags-of-popcorn)
- File: `labeledTrainData.tsv`
- Format: Tab-separated values containing movie reviews and sentiment labels.

### Features:

- id: Unique identifier for each review.
- sentiment: Binary sentiment (0 = negative, 1 = positive).
- review: Text content of the review.

## Project Background

Sentiment analysis is a popular NLP task used in applications like customer feedback analysis, market research, and social media monitoring. This project demonstrates how to preprocess text data and apply machine learning to classify sentiments effectively.

## Concepts

1. **numpy**: Efficient numerical operations for data handling.
2. **pandas**: Data manipulation using DataFrames.
3. **matplotlib.pyplot**: Visualization for data insights.
4. **sklearn.feature_extraction.text**: Tools like CountVectorizer for feature engineering.
5. **CountVectorizer**: Converts text into numerical vectors for machine learning.
6. **sklearn.ensemble**: Includes Random Forest Classifier for model training.
7. **RandomForestClassifier**: An ensemble learning algorithm for classification tasks.
8. **sklearn.metrics**: Tools like ROC AUC score for performance evaluation.
9. **bs4 (BeautifulSoup)**: Removes HTML tags from text data.
10. **re**: Handles text cleaning via regular expressions.
11. **nltk**: Provides NLP tools, including stopwords.
12. **sklearn.model_selection (train_test_split)**: Splits data into training and testing sets.
13. **nltk.corpus (stopwords)**: Contains common stopwords for exclusion.

### Why Remove HTML Tags?

HTML tags can add noise to text data, affecting the model's ability to learn meaningful patterns.

### Why Remove Punctuation?

Punctuation does not convey sentiment and can interfere with text vectorization.

### Why Convert to Lowercase?

Standardizing text case ensures that words like "Movie" and "movie" are treated the same.

### Why remove stopwords?
Stopwords (e.g., "the", "is", "and") do not carry significant meaning and can dilute the model's focus on sentiment-carrying words.

### Training Details

- **Features (X)**: Cleaned and preprocessed text data.
- **Labels (Y)**: Sentiment labels (positive/negative).  
```python
train_x, test_x, y_train, y_test = train_test_split(x, y, test_size=0.1)
```

### Bag-of-Words Model

- Transforms text into a numerical matrix using word frequencies.
```python
vectorizer = CountVectorizer(max_features=5000)
train_x = vectorizer.fit_transform(train_x).toarray()
```
This method creates a matrix of word frequencies representing the most common 5000 words in the dataset.

### Random Forest Classifier

A Random Forest is an ensemble learning method that combines multiple decision trees for robust classification. The model is trained using RandomForestClassifier(n_estimators=100, random_state=42).

### Vectorizer and Transform

- **Vectorizer**: Creates numerical features from text.
- **Transform**: Applies the learned vocabulary to transform new data.

### Model and Predict

- **Model**: The trained Random Forest Classifier.
- **Predict**: Generates predictions on unseen data.

## License

This project is licensed under the MIT License.

## Author

Developed by Yash Mittal. Version 1.0
