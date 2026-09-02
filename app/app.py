import streamlit as st
import joblib
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# ==========================================
# NLTK Resources
# ==========================================

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


# ==========================================
# Load Saved Model and TF-IDF Vectorizer
# ==========================================

model = joblib.load("models/spam_classifier.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))


# ==========================================
# Text Preprocessing
# ==========================================

def preprocess_text(text):
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📩",
    layout="centered"
)


# ==========================================
# Custom CSS
# ==========================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Header
# ==========================================

st.markdown(
    '<div class="title">📩 Spam Mail Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether a message is <b>Spam</b> or <b>Ham</b> '
    'using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# About Project
# ==========================================

with st.expander("ℹ️ About this project"):

    st.write(
        """
        This Spam Mail Detector uses Natural Language Processing (NLP)
        and Machine Learning to classify messages.

        **Model:** Multinomial Naive Bayes

        **Feature Extraction:** TF-IDF

        **Dataset:** SMS Spam Collection

        **Evaluation Accuracy:** 96.68%
        """
    )


# ==========================================
# Disclaimer
# ==========================================

st.info(
    "💡 This tool uses machine learning to make predictions "
    "and may occasionally misclassify messages."
)


# ==========================================
# Message Input
# ==========================================

st.subheader("📝 Enter your message")

message = st.text_area(
    "Message",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# ==========================================
# Prediction
# ==========================================

if st.button("🔍 Check Message", use_container_width=True):

    # Check empty input
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        # --------------------------------------
        # Preprocess message
        # --------------------------------------

        processed_message = preprocess_text(message)


        # --------------------------------------
        # Show processed text
        # --------------------------------------

        with st.expander("🔎 View processed text"):
            st.write(processed_message)


        # --------------------------------------
        # Convert text into TF-IDF features
        # --------------------------------------

        message_vector = tfidf.transform(
            [processed_message]
        )


        # --------------------------------------
        # Make prediction
        # --------------------------------------

        prediction = model.predict(
            message_vector
        )[0]


        # --------------------------------------
        # Get probabilities
        # --------------------------------------

        probabilities = model.predict_proba(
            message_vector
        )[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )

        confidence = (
            probability_dict[prediction] * 100
        )


        # ======================================
        # Display Result
        # ======================================

        if prediction == "spam":

            st.error(
                f"🚨 SPAM MESSAGE\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.success(
                f"✅ HAM MESSAGE\n\n"
                f"Confidence: {confidence:.2f}%"
            )


        # ======================================
        # Prediction Details
        # ======================================

        st.subheader("📊 Prediction Details")

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Ham Probability",
                f"{probability_dict['ham'] * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Spam Probability",
                f"{probability_dict['spam'] * 100:.2f}%"
            )


# ==========================================
# Footer
# ==========================================

st.markdown(
    '<div class="footer">'
    'Built using Python • NLP • TF-IDF • '
    'Multinomial Naive Bayes • Streamlit'
    '</div>',
    unsafe_allow_html=True
)