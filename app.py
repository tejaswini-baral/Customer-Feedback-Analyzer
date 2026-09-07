import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sentiment import train_model, predict_sentiment
from topic_model import train_topic_model


st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Customer Feedback Analyzer")

st.write(
    "Analyze customer feedback using Machine Learning "
    "and classify reviews as Positive, Neutral, or Negative."
)

st.divider()


@st.cache_data
def load_data():
    return pd.read_csv("data/Clothing_Reviews.csv")


df = load_data()


@st.cache_resource
def load_model(data):
    return train_model(data)


model, vectorizer = load_model(df)


@st.cache_resource
def load_topic_model(data):
    topic_texts = data["Review Text"].dropna().astype(str).tolist()
    return train_topic_model(topic_texts, n_topics=5)


topic_model, topic_vectorizer = load_topic_model(df)


def get_topic_name(words):
    words = set(word.lower() for word in words)

    topic_keywords = {
        "🚚 Delivery & Shipping": {
            "delivery", "shipping", "ship", "package",
            "order", "arrived", "arrival", "mail"
        },

        "✨ Product Quality": {
            "quality", "fabric", "material", "soft",
            "comfortable", "durable", "well", "made"
        },

        "📏 Size & Fit": {
            "size", "fit", "small", "large", "tight",
            "loose", "length", "true"
        },

        "👗 Style & Appearance": {
            "dress", "style", "color", "colour",
            "design", "look", "beautiful", "pretty"
        },

        "💰 Price & Value": {
            "price", "worth", "money", "expensive",
            "cheap", "cost", "value", "sale"
        }
    }

    best_topic = "🛍️ Product & Shopping Experience"
    best_score = 0

    for topic_name, keywords in topic_keywords.items():
        score = len(words.intersection(keywords))

        if score > best_score:
            best_score = score
            best_topic = topic_name

    return best_topic


def predict_topic(review):
    review_vector = topic_vectorizer.transform([str(review)])
    topic_probabilities = topic_model.transform(review_vector)[0]

    dominant_topic_index = topic_probabilities.argmax()

    topic_words = topic_model.components_[dominant_topic_index]
    feature_names = topic_vectorizer.get_feature_names_out()

    top_words = [
        feature_names[i]
        for i in topic_words.argsort()[-8:][::-1]
    ]

    return get_topic_name(top_words)


st.sidebar.header("📊 Dashboard")

page = st.sidebar.radio(
    "Choose an option:",
    [
        "Analyze Single Review",
        "Analyze Multiple Reviews",
        "Dataset Overview",
        "Sentiment Distribution"
    ]
)


if page == "Analyze Single Review":

    st.header("🔍 Analyze Single Review")

    review = st.text_area(
        "Enter a customer review:",
        placeholder="Example: The product quality is excellent!",
        height=150
    )

    if st.button("Analyze Review", type="primary"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:
            prediction, confidence, probabilities = predict_sentiment(
                review,
                model,
                vectorizer
            )

            st.subheader("Analysis Result")

            if prediction == "Positive":
                st.success(f"😊 Sentiment: {prediction}")

            elif prediction == "Negative":
                st.error(f"😞 Sentiment: {prediction}")

            else:
                st.warning(f"😐 Sentiment: {prediction}")

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.divider()

            topic_name = predict_topic(review)

            st.subheader("🧠 Topic Analysis")
            st.success(f"🎯 Main Topic: {topic_name}")

            st.divider()

            st.subheader("📈 Sentiment Probability")

            probability_df = pd.DataFrame(
                {
                    "Sentiment": probabilities.keys(),
                    "Probability": [
                        value * 100
                        for value in probabilities.values()
                    ]
                }
            )

            st.bar_chart(
                probability_df.set_index("Sentiment")
            )


elif page == "Analyze Multiple Reviews":

    st.header("📁 Analyze Multiple Reviews")

    st.write(
        "Upload a CSV file containing multiple customer reviews. "
        "The system will automatically analyze every review."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            reviews_df = pd.read_csv(uploaded_file)

            st.subheader("📋 Uploaded Data")
            st.dataframe(
                reviews_df.head(10),
                use_container_width=True
            )

            possible_columns = [
                "Review",
                "review",
                "Review Text",
                "review_text",
                "Review_Text",
                "Text",
                "text",
                "Feedback",
                "feedback"
            ]

            review_column = None

            for column in possible_columns:
                if column in reviews_df.columns:
                    review_column = column
                    break

            if review_column is None:

                st.info(
                    "The review column could not be detected automatically."
                )

                review_column = st.selectbox(
                    "Select the column containing reviews:",
                    reviews_df.columns
                )

            st.success(
                f"Review column selected: **{review_column}**"
            )

            if st.button(
                "Analyze All Reviews",
                type="primary"
            ):

                analysis_df = reviews_df.dropna(
                    subset=[review_column]
                ).copy()

                if len(analysis_df) == 0:

                    st.error(
                        "No valid reviews were found in the uploaded file."
                    )

                else:

                    progress = st.progress(0)

                    predictions = []
                    confidences = []
                    topics = []

                    total = len(analysis_df)

                    for i, review in enumerate(
                        analysis_df[review_column]
                    ):

                        prediction, confidence, _ = predict_sentiment(
                            review,
                            model,
                            vectorizer
                        )

                        predictions.append(prediction)
                        confidences.append(confidence * 100)

                        topic = predict_topic(review)
                        topics.append(topic)

                        progress.progress(
                            (i + 1) / total
                        )

                    analysis_df["Sentiment"] = predictions

                    analysis_df["Confidence (%)"] = [
                        round(value, 2)
                        for value in confidences
                    ]

                    analysis_df["Main Topic"] = topics

                    st.success(
                        f"Successfully analyzed {len(analysis_df)} reviews!"
                    )

                    st.divider()

                    st.subheader("📊 Analysis Summary")

                    sentiment_counts = (
                        analysis_df["Sentiment"].value_counts()
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "😊 Positive",
                        sentiment_counts.get("Positive", 0)
                    )

                    col2.metric(
                        "😐 Neutral",
                        sentiment_counts.get("Neutral", 0)
                    )

                    col3.metric(
                        "😞 Negative",
                        sentiment_counts.get("Negative", 0)
                    )

                    st.divider()

                    st.subheader("🧠 Topic Summary")

                    topic_counts = (
                        analysis_df["Main Topic"].value_counts()
                    )

                    st.bar_chart(topic_counts)

                    st.divider()

                    st.subheader("📋 Review Analysis Results")

                    st.dataframe(
                        analysis_df,
                        use_container_width=True
                    )

                    st.subheader("📈 Sentiment Distribution")

                    st.bar_chart(sentiment_counts)

                    st.subheader("📊 Sentiment Percentage")

                    fig, ax = plt.subplots()

                    ax.pie(
                        sentiment_counts.values,
                        labels=sentiment_counts.index,
                        autopct="%1.1f%%",
                        startangle=90,
                        wedgeprops=dict(width=0.8)
                    )

                    ax.axis("equal")

                    st.pyplot(fig)

                    csv = analysis_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Analyzed CSV",
                        data=csv,
                        file_name="analyzed_reviews.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Unable to process the file: {e}"
            )


elif page == "Dataset Overview":

    st.header("📋 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reviews",
        len(df)
    )

    col2.metric(
        "Total Columns",
        len(df.columns)
    )

    col3.metric(
        "Missing Reviews",
        df["Review Text"].isnull().sum()
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Rating Distribution")

    rating_counts = (
        df["Rating"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(rating_counts)


elif page == "Sentiment Distribution":

    st.header("📊 Sentiment Distribution")

    sentiment_df = df.dropna(
        subset=["Review Text"]
    ).copy()

    sentiment_df["sentiment"] = sentiment_df[
        "Rating"
    ].apply(
        lambda rating:
        "Negative"
        if rating <= 2
        else "Neutral"
        if rating == 3
        else "Positive"
    )

    sentiment_counts = (
        sentiment_df["sentiment"].value_counts()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "😊 Positive",
        sentiment_counts.get("Positive", 0)
    )

    col2.metric(
        "😐 Neutral",
        sentiment_counts.get("Neutral", 0)
    )

    col3.metric(
        "😞 Negative",
        sentiment_counts.get("Negative", 0)
    )

    st.divider()

    st.subheader("Sentiment Count")

    st.bar_chart(sentiment_counts)

    st.subheader("Sentiment Percentage")

    fig, ax = plt.subplots()

    ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)