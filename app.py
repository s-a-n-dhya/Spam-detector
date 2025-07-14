import streamlit as st
import joblib
import re

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# Streamlit UI
st.title("SMS Spam Detector")
st.markdown("Enter a message below to check if it's **SPAM** or **NOT SPAM**.")

user_input = st.text_area("✉ Message Text")

if st.button("Detect Spam"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        st.markdown("### Prediction Confidence:")
        st.write(f"✅ Not Spam: {proba[0]*100:.2f}%")
        st.write(f"❌ Spam: {proba[1]*100:.2f}%")

        if pred == 1:
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("🟢 This message is **NOT SPAM**.")
