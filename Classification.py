#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ali Alzurufi
# Professor Lauren
# Date: November 6 2023
# MCS 5223: Text Mining and Data Analytics


# In[2]:


import sklearn
import string
import nltk
import warnings 

warnings.filterwarnings('ignore')


# In[3]:


import pickle 

with open("X_train.txt", "rb") as fp:
    X_train = pickle.load(fp)

with open("X_test.txt", "rb") as fp:
    X_test = pickle.load(fp)

with open("y_train.txt", "rb") as fp:
    y_train = pickle.load(fp)

with open("y_test.txt", "rb") as fp:
    y_test = pickle.load(fp)    


# In[7]:


# BASELINE

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# TF-IDF Vector
tdf = TfidfVectorizer(stop_words = 'english', max_features=100)
tfidf = tdf.fit(X_train)
#print(tfidf.vocabulary_)
tfidfVectorTrain = tfidf.transform(X_train)
tfidfVectorTest = tfidf.transform(X_test)


# K-Nearest Neighbor
knn = KNeighborsClassifier()

knn.fit(tfidfVectorTrain, y_train)

knn_predict = knn.predict(tfidfVectorTest)

target_names = ['negative', 'positive']
print('K-Nearest Neighbor: ')
print(classification_report(y_test, knn_predict, target_names = target_names))


# Decision Tree
tree = DecisionTreeClassifier()

tree.fit(tfidfVectorTrain, y_train)

tree_predict = tree.predict(tfidfVectorTest)

target_names = ['negative', 'positive']
print("Decision Tree: ")
print(classification_report(y_test, tree_predict, target_names = target_names))


# Multi-Layer Perceptron
mlpTrain = MLPClassifier(alpha = 0.01, hidden_layer_sizes = (20, 1))

mlpModel = mlpTrain.fit(tfidfVectorTrain.toarray(), y_train)

mlp_predict = mlpModel.predict(tfidfVectorTest.toarray())

print("Multi-Layer Perceptron: ")
print(classification_report(y_test, mlp_predict))


# Support Vector Machine
svm = SVC(kernel = 'linear')

svm.fit(tfidfVectorTrain, y_train)

svm_predict = svm.predict(tfidfVectorTest)

print("Support Machine Vector: ")
print(classification_report(y_test, svm_predict, target_names = target_names))


# In[12]:


# BASELINE

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# TF-IDF Vector
tdf = TfidfVectorizer(stop_words = 'english', max_features=200)
tfidf = tdf.fit(X_train)
#print(tfidf.vocabulary_)
tfidfVectorTrain = tfidf.transform(X_train)
tfidfVectorTest = tfidf.transform(X_test)
print()


# K-Nearest Neighbor
knn = KNeighborsClassifier()

knn.fit(tfidfVectorTrain, y_train)

knn_predict = knn.predict(tfidfVectorTest)

target_names = ['negative', 'positive']
print('K-Nearest Neighbor: ')
print(classification_report(y_test, knn_predict, target_names = target_names))


# Decision Tree
tree = DecisionTreeClassifier()

tree.fit(tfidfVectorTrain, y_train)

tree_predict = tree.predict(tfidfVectorTest)

target_names = ['negative', 'positive']
print('Decision Tree: ')
print(classification_report(y_test, tree_predict, target_names = target_names))


# Multi-Layer Perceptron
mlpTrain = MLPClassifier(alpha = 0.01, hidden_layer_sizes = (20, 1))

mlpModel = mlpTrain.fit(tfidfVectorTrain.toarray(), y_train)

mlp_predict = mlpModel.predict(tfidfVectorTest.toarray())

print('Multi-Layer Perceptron: ')
print(classification_report(y_test, mlp_predict))


# Support Vector Machine
svm = SVC(kernel = 'linear')

svm.fit(tfidfVectorTrain, y_train)

svm_predict = svm.predict(tfidfVectorTest)

print('Support Vector Machine: ')
print(classification_report(y_test, svm_predict, target_names = target_names))


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re


# Text Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

X_train = [preprocess_text(x) for x in X_train]
X_test = [preprocess_text(x) for x in X_test]

# TF-IDF Vector
tdf = TfidfVectorizer(stop_words = 'english', max_features=200)
tfidf = tdf.fit(X_train)
tfidfVectorTrain = tfidf.transform(X_train)
tfidfVectorTest = tfidf.transform(X_test)
print()


# K-Nearest Neighbor
knn = KNeighborsClassifier()

knn.fit(tfidfVectorTrain, y_train)

knn_predict = knn.predict(tfidfVectorTest)

target_names = ['negative', 'positive']
print('K-Nearest Neighbor: ')
print(classification_report(y_test, knn_predict, target_names = target_names))


# Decision Tree
tree = DecisionTreeClassifier()

tree.fit(tfidfVectorTrain, y_train)

tree_predict = tree.predict(tfidfVectorTest)

target_names = ['negative', 'positive']
print('Decision Tree: ')
print(classification_report(y_test, tree_predict, target_names = target_names))


# Multi-Layer Perceptron
mlpTrain = MLPClassifier(alpha = 0.01, hidden_layer_sizes = (20, 1))

mlpModel = mlpTrain.fit(tfidfVectorTrain.toarray(), y_train)

mlp_predict = mlpModel.predict(tfidfVectorTest.toarray())

print('Multi-Layer Perceptron: ')
print(classification_report(y_test, mlp_predict))


# In[8]:


# The F1 score seemed to improve slightly but after trying several methods, it did not go up 3%
# Some preprocessing techqniques did hurt the score more than others so
# it was more of a trial and error approach to determine which methods
# produced a higher F1 score


# In[ ]:


# I have neither given nor recieved unauthorized aid in completing this work, nor have I presented
# someone else's work as my own.

