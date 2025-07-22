import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download stopwords (only once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load dataset
df = pd.read_csv('large_phishing_dataset.csv')

# Preprocess emails
df['email_clean'] = df['email'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['email_clean'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Prediction function
def predict_email(email_text):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "⚠️ Phishing Email" if prediction[0] == 1 else "✅ Legitimate Email"

# ----------------------
# Streamlit App Layout
# ----------------------

st.set_page_config(page_title="Phishing Email Detector", layout="centered")

st.title("📧 Phishing Email Detection System")
st.write("Enter an email message below to check if it's phishing or legitimate.")

email_input = st.text_area("✉️ Email Content", height=200)

if st.button("Check Email"):
    if not email_input.strip():
        st.warning("Please enter email content to analyze.")
    else:
        result = predict_email(email_input)
        st.success(f"Result: {result}")
