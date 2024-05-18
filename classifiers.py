import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load your dataset
df = pd.read_csv('C:/Users/Guest User/OneDrive/Desktop/sentiment_analysis/tripadvisor_hotel_reviews.csv')  # Replace with your dataset path

# Display the first few rows of the dataset
print(df.head())

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment
def get_sentiment_label(review):
    score = sia.polarity_scores(review)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to each review
df['label'] = df['Review'].apply(get_sentiment_label)

# Map labels to numerical values
label_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
df['label'] = df['label'].map(label_mapping)

# Display the labeled data
print(df.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['label'], test_size=0.2, random_state=42)

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

# Print evaluation metrics for logistic regression model
print("Metrics for Logistic Regression Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_mapping.keys()))


# Initialize the Decision Tree
model1 = DecisionTreeClassifier()

# Train the model
model1.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred1 = model1.predict(X_test_tfidf)

# Print evaluation metrics for Decision Tree
print("Metrics for Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Classification Report:\n", classification_report(y_test, y_pred1, target_names=label_mapping.keys()))


# Initialize the Voting Classifier
model2 = DecisionTreeClassifier()

# Train the model
model2.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred2 = model2.predict(X_test_tfidf)

# Print evaluation metrics for Voting Classifier
print("Metrics for Voting Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Classification Report:\n", classification_report(y_test, y_pred2, target_names=label_mapping.keys()))


# Initialize the Support Vector Machine
model3 = SVC()

# Train the model
model3.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred3 = model3.predict(X_test_tfidf)

# Print evaluation metrics for Support Vector Machine
print("Metrics for  Support Vector Machine")
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Classification Report:\n", classification_report(y_test, y_pred3, target_names=label_mapping.keys()))

# Initialize the Multi-Nomibal Naive Bayes
model4 = MultinomialNB()

# Train the model
model4.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred4 = model4.predict(X_test_tfidf)

# Print evaluation metrics for Multi-Nomibal Naive Bayes
print("Metrics for  Multi-Nomibal Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred4))
print("Classification Report:\n", classification_report(y_test, y_pred4, target_names=label_mapping.keys()))

