import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords the first time only
nltk.download('stopwords')


# Load dataset
data = pd.read_csv('data/phishing_dataset.csv')  # update filename if needed

# Show basic info
print("🔍 Dataset Preview:")
print(data.head())

print("\n📊 Dataset Summary:")
print(data.info())

print("\n📈 Label Distribution:")
print(data['label'].value_counts())

# Step 1: Lowercase the email content
data['text_clean'] = data['text_combined'].str.lower()

import re

# Step 2: Remove punctuation, numbers, special characters
data['text_clean'] = data['text_clean'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

stop_words = set(stopwords.words('english'))

# Step 3: Remove stopwords
data['text_clean'] = data['text_clean'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in stop_words)
)


print("\n🧽 Cleaned text (no stopwords):")
print(data['text_clean'].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for speed
X = vectorizer.fit_transform(data['text_clean'])

# Labels
y = data['label']

print("\n📐 TF-IDF Matrix Shape:")
print(X.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 5: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
import joblib
joblib.dump(model, 'phishing_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("✅ Model and vectorizer saved successfully!")

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\n🧠 Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation, numbers, special chars
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def predict_email(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

if __name__ == "__main__":
    while True:
        print("\nEnter an email text to check (or 'exit' to quit):")
        user_input = input()
        if user_input.strip().lower() == 'exit':
            break
        result = predict_email(user_input)
        print(f"Prediction: {result}")
