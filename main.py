import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


data = pd.read_csv('./data/spam.csv', encoding='latin-1').iloc[:, :2].values

le = LabelEncoder()
target = le.fit_transform(data[:, 0])

# data visualization
unique, count = np.unique(target, return_counts=True)
plt.xticks(unique, unique)
plt.bar(unique, count, color=['green', 'blue'])
plt.title("True Plot")
plt.xlabel("Spam")
plt.ylabel("Count")
plt.show()

print('sentence quantity: {} ,train sample: {} ,test sample: {}'.format(len(target), len(target) * 0.8,
                                                                        len(target) * 0.2))

Stopwords = stopwords.words('english')
stemmer = PorterStemmer()
email = data[:, 1]
email = [re.sub("[^a-zA-Z]", " ", e) for e in email]
strings = np.char.split(np.char.lower(email))
words = [[stemmer.stem(word) for word in string if word not in set(Stopwords)] for string in strings]
sentence = [' '.join(row) for row in words]

cv = CountVectorizer()
tfidf = TfidfTransformer()
X = sentence
X = cv.fit_transform(X).toarray()
y = tf.keras.utils.to_categorical(target, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)


