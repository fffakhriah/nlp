import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from utils.preprocessing import preprocess_text

df = pd.read_csv("data/clean_ai_human_prompts_dataset.csv")

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df['processed_text'] = df['clean_text'].apply(preprocess_text)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['generated'])

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['processed_text'])
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pickle.dump(tfidf, open("model/tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("model/logistic_model.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print("Model trained & saved.")
