import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split

nltk.download('vader_lexicon')

# Load your dataset
data = pd.read_csv("C:/Users/Guest User/OneDrive/Desktop/sentiment_analysis/tripadvisor_hotel_reviews.csv")  # Make sure to provide the correct path to your dataset

# Display the first few rows of the dataset
print(data.head())

# Function to label sentiment based on rating
def label_sentiment(rating):
    if rating > 3:
        return 'positive'
    elif rating < 3:
        return 'negative'
    else:
        return 'neutral'

# Apply the labeling function
data['label'] = data['Rating'].apply(label_sentiment)

# Map labels to numerical values
label_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
data['label'] = data['label'].map(label_mapping)

# Display the labeled data
print(data.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['label'], test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred = model.predict(X_test_tfidf)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_mapping.keys()))


