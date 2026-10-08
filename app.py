import streamlit as st
import joblib

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aviation Sentiment Analysis",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }
        .main-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .card {
            background: #ffffff;
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
            border: 1px solid #e2e8f0;
        }
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            font-size: 1rem !important;
            padding: 0.75rem !important;
        }
        .stTextArea textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #cbd5e1;
            background-color: #ffffff;
            color: #0f172a;
            font-weight: 500;
            padding: 0.55rem 0.9rem;
            transition: all 0.15s ease-in-out;
            width: 100%;
        }
        .stButton > button:hover {
            border-color: #2563eb;
            color: #2563eb;
            background-color: #f1f5ff;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            border: none;
            font-weight: 600;
            padding: 0.7rem 1rem;
            font-size: 1rem;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
            color: white;
        }
        .result-box {
            padding: 1.2rem 1.5rem;
            border-radius: 14px;
            margin-top: 1.2rem;
            font-size: 1.1rem;
            font-weight: 600;
            text-align: center;
            border: 1px solid transparent;
        }
        .result-positive {
            background: #ecfdf5;
            color: #065f46;
            border-color: #a7f3d0;
        }
        .result-negative {
            background: #fef2f2;
            color: #991b1b;
            border-color: #fecaca;
        }
        .result-neutral {
            background: #fffbeb;
            color: #92400e;
            border-color: #fde68a;
        }
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    pipe = load_model("sentiment_model.pkl")
except Exception as e:
    st.error(f"⚠️ Could not load `sentiment_model.pkl`: {e}")
    st.stop()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">✈️ Aviation Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analyze passenger reviews and predict the sentiment instantly.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

examples = {
    "Example 1": "The flight was on time, the crew was friendly, and the seats were comfortable. Excellent experience overall!",
    "Example 2": "The flight departed on time but the food was average and the entertainment system was limited.",
    "Example 3": "My luggage was lost, the staff was rude, and the flight was delayed for over three hours. Very disappointing.",
}

# ---------------------------------------------------------
# Card layout
# ---------------------------------------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("##### Try an example review")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Example 1", use_container_width=True):
            st.session_state.review_text = examples["Example 1"]
    with col2:
        if st.button("Example 2", use_container_width=True):
            st.session_state.review_text = examples["Example 2"]
    with col3:
        if st.button("Example 3", use_container_width=True):
            st.session_state.review_text = examples["Example 3"]

    st.markdown("##### Enter your review")
    review = st.text_area(
        label="Passenger review",
        value=st.session_state.review_text,
        placeholder="Enter passenger review",
        height=150,
        label_visibility="collapsed",
    )
    st.session_state.review_text = review

    predict_clicked = st.button("🔍 Predict Sentiment", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_clicked:
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                pred = pipe.predict([review])[0]
                label = str(pred).strip().lower()

                if label in ("positive", "pos", "1"):
                    css_class, emoji, text = "result-positive", "😊", "Positive"
                elif label in ("negative", "neg", "0"):
                    css_class, emoji, text = "result-negative", "😞", "Negative"
                else:
                    css_class, emoji, text = "result-neutral", "😐", "Neutral"

                st.markdown(
                    f'<div class="result-box {css_class}">{emoji} Predicted Sentiment: {text}</div>',
                    unsafe_allow_html=True,
                )

                if hasattr(pipe, "predict_proba"):
                    try:
                        probs = pipe.predict_proba([review])[0]
                        classes = pipe.classes_
                        prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
                        st.markdown("###### Confidence")
                        for c, p in prob_dict.items():
                            st.progress(min(max(p, 0.0), 1.0), text=f"{c}: {p*100:.1f}%")
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown('<div class="footer">Built with Streamlit • Aviation Sentiment Analysis</div>', unsafe_allow_html=True)