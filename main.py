# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('IMDB/labeledTrainData.tsv', delimiter="\t", quoting=3)
print("Read labeled Train Data to df")

# Download NLTK stopwords
nltk.download('stopwords')
print("Downloaded stopwords from Natural Language Toolkit (nltk)")

# Text preprocessing function
def process(review):
    review = BeautifulSoup(review).get_text()  # Remove HTML tags
    review = re.sub("[^a-zA-Z]", ' ', review)  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Split into words
    swords = set(stopwords.words("english"))  # Get stopwords
    review = [w for w in review if w not in swords]  # Remove stopwords
    return " ".join(review)  # Join the words back together

# Apply the preprocessing to the entire dataset
train_x_tum = []
for r in range(len(df["review"])):
    if (r + 1) % 1000 == 0:
        print("No of reviews processed =", r + 1)
    train_x_tum.append(process(df["review"][r]))

# Prepare the data for training
x = train_x_tum
y = np.array(df["sentiment"])

# Split the data into training and testing sets
train_x, test_x, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
print("Split the data into training and testing sets")

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
train_x = vectorizer.fit_transform(train_x).toarray()
print("Vectorized the training data into feature vectors")

# Train the RandomForestClassifier
print("Training the RandomForestClassifier model")
model = RandomForestClassifier(n_estimators=100, random_state=42)
# A RandomForestClassifier object is created with two parameters:
# `n_estimators=100`: This specifies that the random forest will consist of 100 decision trees. More trees can lead to better performance but also increase computation time.
# `random_state=42`: This is a seed for the random number generator, ensuring that the results are reproducible. Using the same seed will yield the same results across different runs.
model.fit(train_x, y_train)
# The fit method is called on the model, which trains the random forest using the training data.
# `train_x`: This is the feature matrix created from the training data, where each row corresponds to a review and each column corresponds to a feature (word) extracted by the `CountVectorizer`.
# `y_train`: This is the target variable (sentiment labels) corresponding to the training data, indicating whether each review is positive or negative.

# Convert the test data to feature vectors
test_xx = vectorizer.transform(test_x).toarray()
print("Transformed the test data into feature vectors")

# Make predictions on the test data
test_predict = model.predict(test_xx)
print("Made predictions on the test data")

# Evaluate the model's performance using ROC AUC score
acc = roc_auc_score(y_test, test_predict)
print("Accuracy: %", acc * 100)
