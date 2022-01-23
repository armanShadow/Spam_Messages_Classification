from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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

print('sentence quantity: {} ,train sample: {} ,test sample: {}'.format(len(target),len(target)*0.8,len(target)*0.2))

