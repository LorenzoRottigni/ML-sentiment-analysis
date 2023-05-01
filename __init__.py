""" ML Sentiment Analysis

src: https://raw.githubusercontent.com/pieroit/corso_ml_python_youtube_pollo/master/movie_review.csv
From dataset movie_review train the model to unsderstand if a review is positive or negative.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Bernoulli Naive Bayes classifier is from years 80s and was used for spam detection
# It's based on probability and it's a supervised learning algorithm
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
# Library to parse and process tabular data
import pandas as pd

# Dataframe naming convetion
df = pd.read_csv('movie_review.csv', sep=',')

print(df.head())

X = df['text']
y = df['tag']

# Text vectorizer (bag of words)
vect = CountVectorizer(
    # Instead of working with single words, like this
    # it works with both single words and pairs of words
    ngram_range=(1,2)
)

# Overwrite entry matrix X with the vectorized version
X = vect.fit_transform(X)

# Revert the vectorized version to the original text losing
# the order of the words as expected from the bag of words model
print(vect.inverse_transform(X[:2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

model = BernoulliNB()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print('Train accuracy: ', acc_train)
print('Test accuracy: ', acc_test)
