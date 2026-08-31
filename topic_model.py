from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
def train_topic_model(texts, n_topics=5):
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words="english",
        min_df=5
    )
    X = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="online",
        max_iter=3,
        batch_size=256
    )
    lda_model.fit(X)
    return lda_model, vectorizer
def get_topics(model, vectorizer, n_words=8):
    words = vectorizer.get_feature_names_out()
    topics = {}
    for topic_index, topic in enumerate(model.components_):
        top_words = [
            words[i]
            for i in topic.argsort()[-n_words:][::-1]
        ]
        topics[f"Topic {topic_index + 1}"] = top_words
    return topics
def get_topic_distribution(texts, model, vectorizer):
    X = vectorizer.transform(texts)
    return model.transform(X)