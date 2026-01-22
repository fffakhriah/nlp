from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = "ai_text_detector.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError("Model or vectorizer not found. Make sure .pkl files are in the folder.")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip() != "":
            text_vector = vectorizer.transform([text])
            prob = model.predict_proba(text_vector)[0]
            prediction = "AI-Generated" if prob[1] > 0.5 else "Human-Written"
            confidence = round(max(prob) * 100, 2)
        else:
            prediction = "No text entered"
            confidence = 0

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
