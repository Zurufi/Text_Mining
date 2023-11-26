#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ali Alzurufi
# Professor Lauren
# Date: November 6 2023
# MCS 5223: Text Mining and Data Analytics


# In[9]:


import nltk
import gensim
from gensim import models
import warnings
warnings.filterwarnings('ignore')
# nltk.download('gutenberg')


# In[11]:


from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

cleaned_text = []

for text_name in text_names:
    text = gutenberg.raw(text_name).lower()
    sentences = sent_tokenize(text)

    for sentence in sentences:
        words = word_tokenize(sentence)

        # Remove stop words and punctuation
        words = [word for word in words if word not in stop_words and word not in punctuation]

        # Apply lemmatization to each word
        words = [lemmatizer.lemmatize(word) for word in words]

        cleaned_text.append(words)


# In[21]:


from gensim.models import Word2Vec

model = Word2Vec(sentences = cleaned_text, vector_size = 100)

model.save("gutenberg_word2vec.model")


# In[33]:


similar_words = model.wv.most_similar("white", topn = 10)

print(similar_words)


# In[34]:


# Task 6

# The word similarity worked pretty well for the most part but because of the nature of the 
# gutenberg dataset, some of the words were not similar. This is probably because this dataset
# contains old literary works and there can be rare instances of some words that
# are not typically used in modern times. Word2vec also probably wouldn't fully understand the multiple
# meanings of certain words and would produce inaccurate results.


# In[ ]:


# I have neither given nor recieved unauthorized aid in completing this work, nor have I presented
# someone else's work as my own.

