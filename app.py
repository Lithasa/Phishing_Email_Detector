import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("phishing_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Predict function
def predict_email(text):
    cleaned = preprocess_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0][pred]
    label = "Phishing 🚨" if pred == 1 else "Legitimate ✅"
    return label, prob, pred

# Set Streamlit layout
st.set_page_config(page_title="Phishing Email Detector", page_icon="📧", layout="wide")

# Centered layout
col1, col2, col3 = st.columns([1, 2.5, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>📧 Phishing Email Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Paste an email below to check if it's <b>Phishing</b> or <b>Legitimate</b>.</p>", unsafe_allow_html=True)

    email_text = st.text_area("✉️ Email Content:", height=250, label_visibility="visible")

    if st.button("🔍 Predict"):
        if not email_text.strip():
            st.warning("Please enter an email to analyze.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence, pred = predict_email(email_text)

            if pred == 1:
                st.markdown(f"<h3 style='color: red; text-align: center;'>{label}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2%}</b></p>", unsafe_allow_html=True)

                st.error("🚨 This email appears to be a phishing attempt. Be cautious!")

                st.markdown("### 🔒 What you should do:")
                st.markdown("""
                - ❌ Do **not** click any links or download attachments.
                - 🛡️ Report this email to your organization's IT or security team.
                - 🧹 Delete the email from your inbox and trash.
                """)

                st.info("💡 Tip: Phishing emails often create a sense of urgency to trick you into taking quick action.")
            
            else:
                st.markdown(f"<h3 style='color: green; text-align: center;'>{label}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2%}</b></p>", unsafe_allow_html=True)

                st.success("✅ This email looks safe. No suspicious content detected.")
                st.markdown("### ✅ Why this looks legitimate:")
                st.markdown("""
                - No suspicious or urgent language detected.
                - No links asking for personal or login information.
                - Tone and format appear normal and professional.
                """)
