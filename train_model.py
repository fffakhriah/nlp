import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/clean_ai_human_prompts_dataset.csv")

# Basic cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['clean_text'] = df['clean_text'].str.lower().str.strip()
df['generated'] = df['generated'].astype(str).str.strip()

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['generated'])  # AI=0, Human=1

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "model/lr_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(le, "model/label_encoder.pkl")
