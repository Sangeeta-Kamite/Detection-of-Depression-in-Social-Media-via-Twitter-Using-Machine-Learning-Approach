# Detection-of-Depression-in-Social-Media-via-Twitter-Using-Machine-Learning-Approach
This project aims to detect depression-related tweets using Twitter API and Machine Learning techniques. By analyzing syntactical markers in social media posts, the system predicts whether a tweet reflects depressive tendencies or positive emotions. The model is trained using a Random Forest Classifier for high accuracy in sentiment classification.
Features
1. Fetches real-time tweets using Twitter API
2. Preprocess text data(removes stopwords, URLs, special characters)
3. Extract features using TF-IDF Vectorization
4. Classifies tweets into depressive or positive categories
5. Provides a predictive model for mental health insights
Technologies Used
1. Python
2. Tweepy(Twitter API v2)
3. Pandas & Numpy
4. NLTK(Natural Language Processing)
5. Scikit-Learn(Machine Learning)
How it works
1. Fetch Data - Retrieves tweets based on depression-related and positive keywords.
2. Preprocessing - Cleans text by removing noise and stopwords and applying tokenization.
3. Feature Extraction - Converts text into numerical data using TF-TDF Vectorizaer.
4. Model Training - Uses Random Forest Classifier for classification.
5. Prediction – Detects depression-related tweets from new text inputs.
