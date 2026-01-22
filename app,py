import streamlit as st
import pickle
import numpy as np
from utils.preprocessing import preprocess_text

# Load model
tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("model/logistic_model.pkl", "rb"))
le = pickle.load(open("model/label_encoder.pkl", "rb"))

st.set_page_config(page_title="AI Text Detector", layout="centered")

st.title("🧠 AI vs Human Text Detection")
st.write(
    "This application detects whether a given text is **AI-generated** "
    "or **Human-written** using TF-IDF and Logistic Regression."
)

user_text = st.text_area("Enter text:", height=200)

if st.button("Detect"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess_text(user_text)

        if len(cleaned_text.split()) < 5:
            st.warning("Text too short for reliable prediction.")
        else:
            vector = tfidf.transform([cleaned_text])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]

            label = le.inverse_transform([prediction])[0]
            confidence = np.max(probability)

            st.subheader("🔍 Prediction Result")
            st.success(f"Prediction: {label}")
            st.info(f"Confidence Score: {confidence:.2f}")
            st.progress(confidence)
