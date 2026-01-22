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
st.write("Detect whether a text is AI-generated or human-written.")

user_text = st.text_area("Enter text:", height=200)

if st.button("Detect"):
    if user_text.strip() == "":
        st.warning("Please enter text.")
    else:
        cleaned = preprocess_text(user_text)
        vectorized = tfidf.transform([cleaned])
        pred = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        label = le.inverse_transform([pred])[0]
        confidence = np.max(prob)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}")
