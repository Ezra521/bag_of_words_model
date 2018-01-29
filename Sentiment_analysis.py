import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import nltk
#nltk.download()
from nltk.corpus import stopwords


datafile = os.path.join('.', 'data', 'labeledTrainData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
stopwords = {}.fromkeys([ line.rstrip() for line in open('./stopwords.txt')])
raw_example = df['review'][1]
eng_stopwords = set(stopwords)

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)

clean_text(raw_example)
df['clean_review'] = df.review.apply(clean_text)
vectorizer = CountVectorizer(max_features = 5000)
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()


forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, df.sentiment)

confusion_matrix(df.sentiment, forest.predict(train_data_features))

del df
del train_data_features

datafile = os.path.join('.', 'data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
df.head()

test_data_features = vectorizer.transform(df.clean_review).toarray()
test_data_features.shape

result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})

output.to_csv(os.path.join('.', 'data', 'Bag_of_Words_model.csv'), index=False)
del df
del test_data_features