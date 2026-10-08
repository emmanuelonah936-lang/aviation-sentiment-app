import streamlit as st
import joblib

pipe = joblib.load("sentiment_model.pkl")
st.title("Aviation Sentiment Analysis")

text = st.text_area("Enter passanger review:")

if st.button("Predict"):
    result = pipe.predict([text])[0]
    st.success(f"Sentiment: {result}")