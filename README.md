# 💬 Customer Feedback Analyzer

A Machine Learning-based web application that analyzes customer reviews and classifies them into **Positive, Neutral, or Negative** sentiment using Natural Language Processing (NLP).

## 🚀 Features

- Analyze a single customer review.
- Upload a CSV file and analyze multiple reviews at once.
- Sentiment prediction with confidence score.
- Sentiment probability visualization.
- Topic analysis using LDA.
- Identifies major feedback topics such as:
  - 🚚 Delivery & Shipping
  - ✨ Product Quality
  - 📏 Size & Fit
  - 👗 Style & Appearance
  - 💰 Price & Value
- Sentiment distribution using bar charts and pie/donut charts.
- Dataset overview with rating distribution.
- Download analyzed results as a CSV file.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- NLTK

## 🤖 Machine Learning & NLP

### Sentiment Analysis

The application uses the following pipeline:

1. **Text Preprocessing**
   - Converts text to lowercase.
   - Removes URLs and HTML tags.
   - Removes unwanted characters and extra spaces.
   - Tokenizes the review text.

2. **TF-IDF Vectorization**
   - Converts cleaned customer reviews into numerical features.
   - Uses unigrams and bigrams.
   - Uses a maximum of 20,000 features.

3. **Logistic Regression**
   - Classifies reviews into:
     - Positive
     - Neutral
     - Negative
   - Provides confidence scores and class probabilities.

### Final Model Configuration

The final application uses the best-performing configuration selected during model evaluation:

- TF-IDF with unigrams and bigrams `(1,2)`
- Maximum features: `20,000`
- Sublinear TF enabled
- Maximum document frequency: `0.98`
- Logistic Regression
- `C = 2.0`
- Unbalanced class weights

The application directly trains this selected configuration at startup instead of testing multiple configurations, which reduces application startup time.

## 🧠 Topic Analysis

**LDA (Latent Dirichlet Allocation)** is used for topic modeling.

The dominant latent topic is identified from the review, and its important words are mapped to meaningful customer-feedback categories such as:

- Delivery & Shipping
- Product Quality
- Size & Fit
- Style & Appearance
- Price & Value

## 📊 Model Evaluation

The sentiment model was evaluated using a separate validation set for model selection and a held-out test set for final evaluation.

### Performance Results

| Metric | Score |
|---|---:|
| Validation Accuracy | 83.22% |
| Test Accuracy | 83.13% |
| Weighted Precision | 80.38% |
| Weighted Recall | 83.13% |
| Weighted F1 Score | 81.17% |
| Macro Precision | 66.06% |
| Macro Recall | 57.14% |
| Macro F1 Score | 60.34% |

### Confusion Matrix

```text
[[ 220  113  141]
 [  94  158  313]
 [  28   75 3387]]
 The model performs strongly on the Positive class. The lower Macro F1 reflects the class imbalance in the dataset, particularly the weaker performance on the Neutral class.
 
## 📂 Dataset

The project uses the:
Women's Clothing E-Commerce Reviews Dataset (Clothing_Reviews.csv)
The dataset contains customer reviews and ratings that are used for sentiment classification and topic analysis.

### ▶️ How to Run the Project

1. Install dependencies
Open a terminal in the project folder and run:
    pip install -r requirements.txt
2. Start the Streamlit application
    streamlit run app.py
3. Open the application
After starting Streamlit, open the local URL shown in the terminal, normally:
http://localhost:8501

### 📁 Project Structure

Customer-Feedback-Analyzer/
|__Screenshots
|__.gitignore
├── app.py
├── sentiment.py
├── topic_model.py
├── evaluate.py
├── requirements.txt
├── README.md
│
└── data/
    └── Clothing_Reviews.csv

### 📊 Project Workflow

Customer Review / CSV Upload
            ↓
     Text Preprocessing
            ↓
      TF-IDF Vectorization
            ↓
     Logistic Regression
            ↓
   Sentiment + Confidence
            ↓
       LDA Topic Model
            ↓
     Main Feedback Topic
            ↓
      Streamlit Dashboard
            ↓
 Visualization + CSV Export

### 📌 Project Output

The application:
Predicts customer sentiment.
Provides a confidence score.
Shows sentiment probabilities.
Identifies the main feedback topic.
Analyzes multiple reviews through CSV upload.
Displays sentiment and topic distributions.
Provides dataset statistics.
Allows users to download analyzed results as a CSV file.

## 🔮 Future Scope

Use transformer-based NLP models such as BERT for improved contextual understanding.
Add multilingual customer-review support.
Add aspect-based sentiment analysis.
Store and track feedback over time.
Add real-time feedback monitoring and business insights.
