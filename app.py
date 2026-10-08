import streamlit as st
import joblib

st.set_page_config(page_title="Aviation Sentiment Analysis", page_icon="✈️")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    pipe = load_model("sentiment_model.pkl")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.title("✈️ Aviation Sentiment Analysis")
st.write("Type a passenger review and check the sentiment.")

# Faint example shown as placeholder (users can type over it)
placeholder_text = "e.g. Flight was on time and staff were friendly."

review = st.text_area(
    "Passenger review",
    placeholder=placeholder_text,
    height=120,
    label_visibility="collapsed",
)

if st.button("Predict Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        pred = pipe.predict([review])[0]
        label = str(pred).strip().lower()

        if label in ("positive", "pos", "1"):
            st.success("😊 Positive")
        elif label in ("negative", "neg", "0"):
            st.error("😞 Negative")
        else:
            st.warning("😐 Neutral")