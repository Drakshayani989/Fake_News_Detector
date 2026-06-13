import streamlit as st
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
}

.hero-subtitle {
    color: #6b7280;
    margin-top: 0.5rem;
}

.result-card {
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">📰 Fake News Detector</div>
    <div class="hero-subtitle">
        Machine Learning powered classification of news articles using TF-IDF and Logistic Regression
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Input Section
# -----------------------------
left, right = st.columns([3, 1])

with left:
    news_text = st.text_area(
        "Paste News Article",
        height=300,
        placeholder="Paste a complete news article here for best results..."
    )

with right:
    word_count = len(news_text.split()) if news_text else 0
    char_count = len(news_text) if news_text else 0

    st.metric("Words", word_count)
    st.metric("Characters", char_count)
    st.metric("Model Accuracy", "98.56%")

analyze = st.button("🔍 Analyze Article", use_container_width=True)

# -----------------------------
# Prediction Section
# -----------------------------
if analyze:

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article.")
        st.stop()

    if len(news_text.split()) < 50:
        st.warning(
            "⚠️ This model was trained on full-length articles. Results may be less reliable for short text."
        )

    news_vector = vectorizer.transform([news_text])

    prediction = model.predict(news_vector)
    probability = model.predict_proba(news_vector)

    confidence = float(max(probability[0]) * 100)

    st.divider()
    st.subheader("Analysis Result")

    st.progress(int(confidence))

    if prediction[0] == 1:
        st.success(f"✅ Real News")
    else:
        st.error(f"🚨 Fake News")

    st.write(f"**Confidence Score:** {confidence:.2f}%")

# -----------------------------
# Model Performance
# -----------------------------
st.divider()
st.subheader("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", "98.56%")
c2.metric("Precision", "99%")
c3.metric("Recall", "99%")
c4.metric("F1 Score", "99%")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📌 Project Information")

    st.info("""
    **Dataset**
    ISOT Fake & Real News Dataset

    **Feature Extraction**
    TF-IDF Vectorization

    **Algorithm**
    Logistic Regression
    """)

    st.success("Model trained on 44,000+ news articles")

    st.markdown("### Notes")
    st.write(
        "The model works best on complete news articles rather than short headlines or single sentences."
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Built with Python, Scikit-Learn, Streamlit, TF-IDF and Logistic Regression")