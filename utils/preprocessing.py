import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    doc = nlp(text.lower().strip())
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and token.text not in stop_words
    ]
    return " ".join(tokens)
