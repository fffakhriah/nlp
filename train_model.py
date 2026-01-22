import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils.preprocessing import preprocess_text

# Load dataset
df = pd.read_csv("data/clean_ai_human_prompts_dataset.csv")

# Basic cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Preprocess text
df['processed_text'] = df['clean_text'].apply(preprocess_text)

# Encode label
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['generated'])

# TF-IDF (FINAL VERSION – TAK REDUNDANT)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = tfidf.fit_transform(df['processed_text'])
y = df['label_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate (UNTUK RESULT & ANALYSIS)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(tfidf, open("model/tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("model/logistic_model.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print("✅ Model trained & saved successfully.")
