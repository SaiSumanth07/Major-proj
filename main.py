import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
warnings.filterwarnings("ignore")

# Set page title and layout
st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis on Twitter Data")
# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\DELL\Downloads\code (1)\code\code\Twitter_Data.csv", usecols=['clean_text', 'category'])
    return data.sample(frac=0.5, random_state=2)  # Use only 50% of the data initially

twitter_data = load_data()

# Handle missing values and duplicates
twitter_data.dropna(inplace=True)
twitter_data.drop_duplicates(inplace=True)

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Apply preprocessing
twitter_data['tweet'] = twitter_data['clean_text'].apply(preprocess_text)

# Separate features and labels
X = twitter_data['tweet'].values
y = twitter_data['category'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_test_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Display model comparison
st.header("Model Performance Comparison")
results_df = pd.DataFrame(results).T
st.write(results_df)

# Plot comparison graph
st.subheader("Comparison Graph")
plt.figure(figsize=(10, 6))
results_df.plot(kind='bar', figsize=(12, 6))
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc='upper right')
st.pyplot(plt)
plt.close()

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = models[best_model_name]
st.success(f"The best-performing model is **{best_model_name}** with an accuracy of {results[best_model_name]['Accuracy']:.4f}.")

# Predict sentiment using the best model
st.header("Predict Sentiment of a Twitter Post")
user_input = st.text_area("Enter a Twitter post:")

if user_input:
    processed_input = preprocess_text(user_input)
    input_tfidf = vectorizer.transform([processed_input])
    prediction = best_model.predict(input_tfidf)[0]
    
    sentiment_map = {1: "Positive", -1: "Negative", 0: "Neutral"}
    sentiment = sentiment_map.get(prediction, "Unknown")
    st.write(f"**Predicted Sentiment:** {sentiment}")
