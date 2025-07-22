import tkinter as tk
from tkinter import messagebox
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download stopwords once
nltk.download('stopwords')

# Load stopwords once
stop_words = set(stopwords.words('english'))

# Preprocessing function to clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load dataset
df = pd.read_csv('large_phishing_dataset.csv')

# Clean email text
df['email_clean'] = df['email'].apply(clean_text)

# TF-IDF Vectorization with n-grams and max features for generalization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['email_clean'])
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model with better regularization
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Prediction function
def predict_email(email_text):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Phishing Email ⚠️" if prediction[0] == 1 else "Legitimate Email ✅"

# GUI Setup using Tkinter
def check_email():
    email_text = entry.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Required", "Please enter email text.")
        return
    result = predict_email(email_text)
    messagebox.showinfo("Result", result)

# Create the main window
root = tk.Tk()
root.title("Phishing Email Detection")
root.geometry("500x300")

# Instruction Label
tk.Label(root, text="Enter Email Content:", font=('Arial', 12)).pack(pady=10)

# Email Textbox
entry = tk.Text(root, height=8, width=50)
entry.pack()

# Button to trigger prediction
tk.Button(root, text="Check Email", command=check_email, bg="blue", fg="white").pack(pady=10)

# Start the GUI loop
root.mainloop()
