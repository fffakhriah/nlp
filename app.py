from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# ===============================
# Load trained model & vectorizer
# ===============================
MODEL_PATH = "ai_text_detector.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file not found.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ===============================
# Home route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob_human = 0.0
    prob_ai = 0.0
    confidence_msg = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if text:
            # Transform input text using TF-IDF
            text_vector = vectorizer.transform([text])

            # Predict probabilities
            probabilities = model.predict_proba(text_vector)[0]

            # Label mapping (CONSISTENT WITH REPORT)
            # Human = 0, AI = 1
            prob_human = round(probabilities[0] * 100, 2)
            prob_ai = round(probabilities[1] * 100, 2)

            prediction = "AI-Generated" if prob_ai > prob_human else "Human-Written"

            # Confidence warning (optional but good for demo)
            if abs(prob_ai - prob_human) < 10:
                confidence_msg = "⚠️ Prediction confidence is low."

        else:
            prediction = "No text entered."

    return render_template(
        "index.html",
        prediction=prediction,
        prob_human=prob_human,
        prob_ai=prob_ai,
        confidence_msg=confidence_msg
    )

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
