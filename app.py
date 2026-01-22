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

    # 👉 TAMBAH: simpan input text
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "").strip()

        if input_text:
            # Transform input text using TF-IDF
            text_vector = vectorizer.transform([input_text])

            # Predict probabilities
            probabilities = model.predict_proba(text_vector)[0]

            # Label mapping 
            # Human = 0, AI = 1
            prob_human = round(probabilities[0] * 100, 2)
            prob_ai = round(probabilities[1] * 100, 2)

            if prob_ai >= 60:
                prediction = "AI-Generated"
            elif prob_human >= 60:
                prediction = "Human-Written"
            else:
                prediction = "Uncertain (Low Confidence)"

            # Confidence warning 
            if abs(prob_ai - prob_human) < 10:
                confidence_msg = "⚠️ Prediction confidence is low."

        else:
            prediction = "No text entered."

    # 👉 TAMBAH input_text dalam render_template
    return render_template(
        "index.html",
        prediction=prediction,
        prob_human=prob_human,
        prob_ai=prob_ai,
        confidence_msg=confidence_msg,
        input_text=input_text
    )

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
