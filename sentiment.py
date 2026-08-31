import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download("punkt_tab")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return " ".join(tokens)
def train_model(df):
    df = df.dropna(subset=["Review Text"]).copy()
    def get_sentiment(rating):
        if rating <= 2:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        else:
            return "Positive"
    df["sentiment"] = df["Rating"].apply(get_sentiment)
    df["clean_review"] = df["Review Text"].apply(clean_text)
    df = df[df["clean_review"].str.strip() != ""]
    X = df["clean_review"]
    y = df["sentiment"]
    vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1, 2),min_df=2,sublinear_tf=True)
    X_tfidf = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=2000,class_weight="balanced",C=2)
    model.fit(X_tfidf, y)
    return model, vectorizer
def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    probability_dict = dict(
        zip(model.classes_, probabilities)
    )
    confidence = max(probabilities)
    return prediction, confidence, probability_dict