# Ali Alzurufi
# Professor Lauren
# Date: November 6 2023
# MCS 5223: Text Mining and Data Analytics

import nltk
import gensim
from gensim import models
import warnings
warnings.filterwarnings('ignore')
# nltk.download('gutenberg')


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


from gensim.models import Word2Vec

model = Word2Vec(sentences = cleaned_text, vector_size = 100)

model.save("gutenberg_word2vec.model")

similar_words = model.wv.most_similar("white", topn = 10)

print(similar_words)

