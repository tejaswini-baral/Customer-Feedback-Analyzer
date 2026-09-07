import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Download tokenizer data if needed
nltk.download("punkt_tab", quiet=True)


def clean_text(text):
    """Clean and tokenize review text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return " ".join(tokens)


def get_sentiment(rating):
    """Convert rating into the project's three sentiment classes."""
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def _prepare_data(df):
    """Prepare cleaned reviews and labels."""
    df = df.dropna(subset=["Review Text", "Rating"]).copy()
    df["sentiment"] = df["Rating"].apply(get_sentiment)
    df["clean_review"] = df["Review Text"].apply(clean_text)
    df = df[df["clean_review"].str.strip() != ""]
    return df["clean_review"], df["sentiment"]


def _get_configurations():
    """
    Candidate TF-IDF + Logistic Regression configurations.

    The model is selected using a validation split. The winning
    configuration is then retrained on the complete dataset.
    """
    return [
        {
            "name": "TF-IDF (1,2) + LR C=1",
            "vectorizer": {
                "max_features": 15000,
                "ngram_range": (1, 2),
                "min_df": 1,
                "sublinear_tf": True,
                "max_df": 0.98,
            },
            "model": {
                "C": 1.0,
                "class_weight": "balanced",
            },
        },
        {
            "name": "TF-IDF (1,2) + LR C=2",
            "vectorizer": {
                "max_features": 20000,
                "ngram_range": (1, 2),
                "min_df": 1,
                "sublinear_tf": True,
                "max_df": 0.98,
            },
            "model": {
                "C": 2.0,
                "class_weight": "balanced",
            },
        },
        {
            "name": "TF-IDF (1,2) + LR C=4",
            "vectorizer": {
                "max_features": 20000,
                "ngram_range": (1, 2),
                "min_df": 1,
                "sublinear_tf": True,
                "max_df": 0.98,
            },
            "model": {
                "C": 4.0,
                "class_weight": "balanced",
            },
        },
        {
            "name": "TF-IDF (1,3) + LR C=2",
            "vectorizer": {
                "max_features": 25000,
                "ngram_range": (1, 3),
                "min_df": 1,
                "sublinear_tf": True,
                "max_df": 0.98,
            },
            "model": {
                "C": 2.0,
                "class_weight": "balanced",
            },
        },
        {
            "name": "TF-IDF (1,3) + LR C=4",
            "vectorizer": {
                "max_features": 30000,
                "ngram_range": (1, 3),
                "min_df": 1,
                "sublinear_tf": True,
                "max_df": 0.98,
            },
            "model": {
                "C": 4.0,
                "class_weight": "balanced",
            },
        },
        {
            "name": "TF-IDF (1,2) + LR C=2 (unbalanced)",
            "vectorizer": {
                "max_features": 20000,
                "ngram_range": (1, 2),
                "min_df": 1,
                "sublinear_tf": True,
                "max_df": 0.98,
            },
            "model": {
                "C": 2.0,
                "class_weight": None,
            },
        },
    ]


def _build_model(config):
    """Create a vectorizer and classifier from one configuration."""
    vectorizer = TfidfVectorizer(**config["vectorizer"])
    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        **config["model"],
    )
    return model, vectorizer


def train_model(df):
    """
    Automatically test multiple TF-IDF + Logistic Regression configurations,
    select the best one using a stratified validation set, and finally retrain
    the selected configuration on the complete dataset.

    Returns:
        model, vectorizer
    """
    X, y = _prepare_data(df)

    # Keep a fixed validation split so model comparisons are fair.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    best_accuracy = -1.0
    best_config = None

    print("\nTesting sentiment model configurations...")
    print("-" * 55)

    for config in _get_configurations():
        vectorizer = TfidfVectorizer(**config["vectorizer"])

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_valid_tfidf = vectorizer.transform(X_valid)

        model = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            **config["model"],
        )
        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_valid_tfidf)
        accuracy = accuracy_score(y_valid, predictions)

        print(f"{config['name']:<42} {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = config

    # Retrain the winning configuration on ALL available data.
    final_model, final_vectorizer = _build_model(best_config)
    X_all_tfidf = final_vectorizer.fit_transform(X)
    final_model.fit(X_all_tfidf, y)

    print("-" * 55)
    print(f"Best configuration : {best_config['name']}")
    print(f"Validation accuracy: {best_accuracy:.4f}")
    print("Final model trained on the complete dataset.")

    # Store useful information on the model for display/debugging.
    final_model.best_config_name_ = best_config["name"]
    final_model.validation_accuracy_ = best_accuracy

    return final_model, final_vectorizer

def train_model(df):
    """
    Train the final sentiment model using the best
    configuration selected during evaluation.

    Best configuration:
    TF-IDF (1,2) + Logistic Regression C=2 (unbalanced)
    """

    X, y = _prepare_data(df)

    # Best configuration found during model evaluation
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_df=0.98,
    )

    model = LogisticRegression(
        C=2.0,
        class_weight=None,
        max_iter=3000,
        solver="lbfgs",
    )

    # Transform the complete dataset
    X_tfidf = vectorizer.fit_transform(X)

    # Train the final model
    model.fit(X_tfidf, y)

    # Store information for reference
    model.best_config_name_ = (
        "TF-IDF (1,2) + LR C=2 (unbalanced)"
    )

    print("Sentiment model trained successfully.")
    print(
        "Configuration: "
        "TF-IDF (1,2) + LR C=2 (unbalanced)"
    )

    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment, confidence and class probabilities."""
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    probability_dict = dict(zip(model.classes_, probabilities))
    confidence = max(probabilities)

    return prediction, confidence, probability_dict
