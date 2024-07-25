import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict sentiment of a random review
def predict_sentiment(review):
    # Vectorize the review text
    review_vectorized = vectorizer.transform([review])
    # Predict the sentiment
    prediction = model.predict(review_vectorized)
    # Convert the prediction to a label
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return sentiment

# Streamlit app
st.title('Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict its sentiment.')

# Text input for the review
review = st.text_area('Review')

# Predict sentiment
if st.button('Predict Sentiment'):
    sentiment = predict_sentiment(review)
    st.write(f'The predicted sentiment is: {sentiment}')
