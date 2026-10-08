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
st.markdown(
    "<p style='color:#94a3b8; font-size:1rem;'>"
    "Type a passenger review and check the sentiment."
    "</p>",
    unsafe_allow_html=True,
)

# Faint example users can type over
placeholder_text = "e.g. The flight was delayed and baggage lost"

# Make the placeholder fainter
st.markdown(
    """
    <style>
    textarea::placeholder {
        color: #cbd5e1 !important;
        opacity: 1 !important;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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