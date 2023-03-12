# importing the libraries
import pandas as pd
import numpy as np

# importing the dataset
dataset = pd.read_csv('Project2_Sentiment_Analysis/a1_RestaurantReviews_HistoricDump.tsv',delimiter= '\t', quoting=3)

# DATA CLEANING (most imp)
import regex
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

# Corpus - review post data cleaning
corpus=[]

for i in range(0,900):
    review = regex.sub('[^a-zA-Z]',' ', dataset['Review'][i])  #removing spaces fullstops etc
    review = review.lower() #lowercasing everything
    review = review.split() #Splits an input string into an array of substrings
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Data Transformation
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1420)  #got 1420 through hit and trial

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

import pickle
bow_path = 'Project2_Sentiment_Analysis/BOW_Sentiment.pkl'
pickle.dump(cv, open(bow_path,"wb"))

# splitting dataset into training and test
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# fitting ze' Model  (NAIVE BAYES)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# testing the model
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy_score(y_test,y_pred)