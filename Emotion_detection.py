# -*- coding: utf-8 -*-
"""Final_Project_Emotion_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14n7DgQ3x5J8D55ne-BlzwQdB4giS1KsR
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

"""# Loading Data"""

# Upload file code to google collab

from google.colab import files

#this will prompt you to select a file from your local machine
uploaded = files.upload()

import pickle

# Loading the data back
with open('EmotionDetection.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)

#import libraries

"""# Exploratory Data Analysis"""

data.isnull().sum()   #identify null values

data.duplicated().sum() #sum

data.drop_duplicates(inplace=True) #drop any duplicates

data.info()

data   #view data set

data['emotion'].unique()   #giving us the unique class labels for emmotions

#visualization of counts per label (emotion)

data['emotion'].value_counts().plot(kind='bar', color=['magenta', 'cyan', 'pink', 'orange', 'yellow', 'blue'])
plt.xlabel('Emotions')
plt.ylabel('Number of samples')
plt.show()

len(data)

import nltk
nltk.download('punkt')

data['num_of_characters'] = data.text.apply(lambda x:len(x))  #compute the length of characters and add new column

data['num_of_words'] = data['text'].apply(lambda x: len(nltk.word_tokenize(x))) # counting the number of words in each text and adding it as a column.

data['num_of_sentences'] = data['text'].apply(lambda x: len(nltk.sent_tokenize(x))) # counting the number of sentences in each text and adding it as a column.


#summary statistics for the refined dataset
data[['num_of_characters', 'num_of_words', 'num_of_sentences']].describe()

data #viewing data with the new columns for analytics

#density of frequency as a percentage

fig = plt.figure(figsize=(10,6))
sns.kdeplot(x=data['num_of_characters'], hue=data["emotion"])
plt.show()

import nltk
nltk.download('stopwords')

#word clouds as a quick view into the key data per emotion

emotions = data['emotion'].unique()
stopwords = set(nltk.corpus.stopwords.words("english"))

for emotion in emotions:
    text = " ".join(data[data['emotion'] == emotion]['text'])  #transform into string to meet word cloud reqt.
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(text)
    plt.figure(figsize = (4, 4), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.title(emotion)
    plt.show()

"""**Data split for training and test**"""

#create our train set and test set.
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["emotion"], test_size=0.20, random_state=42)

"""**Baseline Numerical Representation: TF-IDF**"""

#cross validation
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf= TfidfVectorizer(max_features=1200)   #generally a 10 to 1 selection (10000 records ideally 1000 features, )

X_train_cv = tfidf.fit_transform(X_train)   #.fit_transform on the training set doing the training
X_test_cv = tfidf.transform(X_test)         # model is done and now apply to test set doing the testing

X_train_cv

"""**Baseline Classifier: Logistic Regression**"""

lr = LogisticRegression()
lr.fit(X_train_cv, y_train)  #applying it to our transformed training set (all numbers and the label)

y_pred_lr = lr.predict(X_test_cv)  # apply the model to the test set and get results


report_lr = classification_report(y_test, y_pred_lr)
print("Classification report of Logistic Regression (Multi-Class):\n", report_lr)

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score


y_pred_lr = lr.predict(X_test_cv)  # apply the model to the test set and get results


report_lr = classification_report(y_test, y_pred_lr)

# Perform 5-fold cross-validation
scores = cross_val_score(lr, X_train_cv, y_train, cv=5)

print(f"Accuracy scores for the 5 folds: {scores}")
print(f"Mean accuracy: {scores.mean()}")

from sklearn.metrics import confusion_matrix
import numpy as np

cm = confusion_matrix(y_test, y_pred_lr, labels=["anger", "fear", "joy", "love", "sadness", "surprise"])
print(cm)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"], yticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

#https://huggingface.co/gpt2
!pip install transformers

#https://pypi.org/project/torch/
!pip install torch torchvision

#import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#load pre-trained model and tokenizer
model_name = "gpt2-medium"
model =  GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

# Initialize the GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

def generate_surprise_sentences(num_sentences=10, max_length=50, temperature=0.49):
    surprise_prompts = ["Surprise! This is", "I am surprised because",
                      "Wow! That is surprising", "I am shocked because",
                      "No way! It turns out that"]
    sentences = []

    for _ in range(num_sentences):
        prompt = random.choice(surprise_prompts)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=100,
                top_p=0.83
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        sentences.append(generated_text)

    return sentences

surprise_sentences = generate_surprise_sentences()

surprise_sentences

#save to google drive

from google.colab import drive
drive.mount('/content/drive')

import pickle

save_path = '/content/drive/My Drive/EmotionDetectionWithTextGen.pkl'

data_to_save = surprise_sentences

with open(save_path, 'wb') as file:
  pickle.dump(data_to_save, file)

#retrieve from google drive

import pickle

#open the file and load the data into google collab from google drive

with open(save_path, 'rb') as file:
  data1 = pickle.load(file)

data1

import pandas as pd

# Create dataframe for generated surprise emotions sentences
df = pd.DataFrame(surprise_sentences, columns=['text'])

df['emotion'] = "surprise"

df
data

# combine to orignal dataset
dataCombo = pd.concat((data, df), ignore_index=True)#, axis=0)


dataCombo

print(len(dataCombo))
print(len(df))
print(len(data))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

# Initialize the GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

def generate_love_sentences(num_sentences=10, max_length=50, temperature=0.49):
    love_prompts = ["I love that because ", "My heart is filled with love for",
                      "Falling in love because", "Love is in the air",
                      "Embrace the love because"]
    sentences = []

    for _ in range(num_sentences):
        prompt = random.choice(love_prompts)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=100,
                top_p=0.83
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        sentences.append(generated_text)

    return sentences

love_sentences = generate_love_sentences()

love_sentences

#save to google drive

from google.colab import drive
drive.mount('/content/drive')

import pickle

save_path = '/content/drive/My Drive/EmotionDetectionWithTextGen.pkl'

data_to_save = love_sentences

with open(save_path, 'wb') as file:
  pickle.dump(data_to_save, file)

#retrieve from google drive

import pickle

#open the file and load the data into google collab from google drive

with open(save_path, 'rb') as file:
  data2 = pickle.load(file)

data2

import pandas as pd

# Create daatframe for generated love emotion sentences

df2 = pd.DataFrame(love_sentences, columns=['text'])

df2['emotion'] = "love"

df2
dataCombo

# Combine second df to dataCombo dataframe
dataCombo = pd.concat([dataCombo, df2], ignore_index=True)

print(len(dataCombo))
print(len(df2))
print(len(data))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()

    # Removal of symbols and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removal of stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Applying the preprocessed text to the dataset
dataCombo['preprocessed_text'] = dataCombo['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataCombo['preprocessed_text'],
    dataCombo['emotion'],
    test_size=0.20,
    random_state=42
)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1800)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# SGD Classifier
sgd_model = SGDClassifier(random_state=42, class_weight='balanced')

# Logistic Regression Classifier
logreg_model = LogisticRegression(random_state=42, class_weight='balanced')

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)

# Combine classifiers using VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('sgd', sgd_model), ('lg', logreg_model), ('gb', gb_model)],
    voting='hard'
)

# Fit the ensemble model on the training data
voting_clf.fit(X_train_tfidf, y_train)

# Predictions on the test set
y_pred_voting = voting_clf.predict(X_test_tfidf)

# Generate and print the classification report for the ensemble model
report_voting = classification_report(y_test, y_pred_voting)
print("Classification Report for Ensemble Model:\n", report_voting)

# Perform 5-fold cross-validation on the ensemble model
ensemble_scores = cross_val_score(voting_clf, X_train_tfidf, y_train, cv=5)

print(f"Accuracy scores for the 5 folds: {ensemble_scores}")
print(f"Mean accuracy: {ensemble_scores.mean()}")


# Confusion Matrix for the ensemble model
cm_voting = confusion_matrix(y_test, y_pred_voting, labels=["anger", "fear", "joy", "love", "sadness", "surprise"])
print("Confusion Matrix for Ensemble Model:")
print(cm_voting)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_voting, annot=True, fmt="d", cmap="Blues", xticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"],
            yticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()