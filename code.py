#import libraries
import nltk
import tweepy
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Authenticate with Twitter API

# Twitter API Credentials
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
ACCESS_TOKEN = "your_access_token"
ACCESS_SECRET = "your_access_secret"

# Authenticate with Twitter
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

#Collect Tweets
def fetch_tweets(query, count=100):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(count):
        tweets.append(tweet.full_text)
    return tweets

# Fetch depression-related tweets
depression_tweets = fetch_tweets("depression OR sad OR hopeless OR anxiety", count=500)
positive_tweets = fetch_tweets("happy OR joy OR excited OR love", count=500)

# Create a dataset
df = pd.DataFrame({
    "text": depression_tweets + positive_tweets,
    "label": [1] * len(depression_tweets) + [0] * len(positive_tweets)  # 1 for depression, 0 for positive
})

#Preprocess the Data
nltk.download("stopwords")
nltk.download("punkt")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"\@w+|\#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuations
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

#Convert Text to Features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"]).toarray()
y = df["label"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Random Forest Model

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict Depression in New Tweets
def predict_depression(tweet):
    cleaned_tweet = clean_text(tweet)
    vectorized_tweet = vectorizer.transform([cleaned_tweet]).toarray()
    prediction = rf_model.predict(vectorized_tweet)
    return "Depressive Tweet" if prediction == 1 else "Positive Tweet"

# Test on a new tweet
sample_tweet = "I feel so hopeless today."
print(predict_depression(sample_tweet))

