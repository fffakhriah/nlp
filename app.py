from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model & vectorizer
MODEL_PATH = "ai_text_detector.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError("Model or vectorizer not found.")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob_human = None
    prob_ai = None

    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip() != "":
            text_vector = vectorizer.transform([text])
            prob = model.predict_proba(text_vector)[0]
            prob_human = round(prob[0]*100, 2)
            prob_ai = round(prob[1]*100, 2)
            prediction = "AI-Generated" if prob_ai > prob_human else "Human-Written"
        else:
            prediction = "No text entered"
            prob_human = 0
            prob_ai = 0

    return render_template(
        "index.html",
        prediction=prediction,
        prob_human=prob_human,
        prob_ai=prob_ai
    )

if __name__ == "__main__":
    app.run(debug=True)
