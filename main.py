import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class LogisticRegressionModel(tf.keras.Model):
    def __init__(self, units, input_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1024, activation='elu', input_dim=input_dim, kernel_initializer='normal')
        self.dense2 = tf.keras.layers.Dense(256, activation='elu', kernel_initializer='normal')
        self.dense3 = tf.keras.layers.Dense(64, activation='elu', kernel_initializer='normal')
        self.dense4 = tf.keras.layers.Dense(16, activation='elu', kernel_initializer='normal')
        self.dense5 = tf.keras.layers.Dense(units=units, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)


def dataVisualization(target):
    # data visualization
    unique, count = np.unique(target, return_counts=True)
    plt.xticks(unique, unique)
    plt.bar(unique, count, color=['green', 'blue'])
    plt.title("True Plot")
    plt.xlabel("Spam")
    plt.ylabel("Count")
    plt.show()


def dataPreprocessing(data, target):
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
    X = tfidf.fit_transform(X).toarray()
    return train_test_split(X, y, test_size=0.20, random_state=12)


def spamMessageClassification(data):
    le = LabelEncoder()
    target = le.fit_transform(data[:, 0])
    print('sentence quantity: {} ,train sample: {} ,test sample: {}'.format(len(target), len(target) * 0.8,
                                                                            len(target) * 0.2))
    dataVisualization(target)
    X_train, X_test, y_train, y_test = dataPreprocessing(data, target)

    model = LogisticRegressionModel(2, X_train.shape[1])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=60)

    y_pred = model.predict(X_test)
    Y_pred = np.argmax(y_pred, axis=1)
    Y_test = np.argmax(y_test, axis=1)

    ac = accuracy_score(Y_pred, Y_test)
    cm = confusion_matrix(Y_pred, Y_test)
    rs = recall_score(Y_pred, Y_test)

    print(f"Confusion Matrix For Test Dataset:\n {cm} \n Accuracy For Test Dataset: {ac} \n "
          f"Recall Score For Test Dataset: {rs}")


if __name__ == '__main__':
    spamMessageClassification(pd.read_csv('./data/spam.csv', encoding='latin-1').iloc[:, :2].values)
