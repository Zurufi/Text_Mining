# Ali Alzurufi
# Professor Lauren
# Date: September 29 2023
# MCS 5223: Text Mining and Data Analytics

""" Description: This programwill read in text from a csv file. It will remove all stopwords, punctuation, usernames, hyperlinks, and non-lexical utterances.
    It will then generate TF-IDF for 200 features, show the vocabulary before and after the cleanup, and print out the matrix as well as its dimensions. """


import pandas as pd
import re
import string
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from autocorrect import Speller
from sklearn.feature_extraction.text import TfidfVectorizer

# read csv file
df = pd.read_csv("SentimentData.csv", encoding="ISO-8859-1")

# non lexical utterances to be removed
non_lexical_utterances = [
    "haha",
    "lol",
    "ha",
    "ya",
    "hey",
    "omg",
    "dis",
    "rt",
    "fr",
    "gon",
    "na",
    "mi",
]

spell = Speller()

# set of english words
english_words = set(words.words())


# Define the preprocess_text function to clean and preprocess text
def preprocess_text(text):
    # removing special characters and digits
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", "", text)

    # removing extra white spaces
    text = re.sub(r"\s+", " ", text).strip()

    # removing URLs
    text = re.sub(r"http\S+", "", text)

    # removing usernames
    text = re.sub(r"@\w+", "", text)

    # remove repeating characters
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # tokenize words
    words = word_tokenize(text)

    # convert words to lowercase and remove punctuation
    words = [word.lower() for word in words if word not in string.punctuation]

    # remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # spell correction
    words = [spell(word) for word in words]

    # remove non english words
    words = [word for word in words if word in english_words]

    # remove non-lexical utterances
    words = [word for word in words if word.lower() not in non_lexical_utterances]

    # revert words to string and return
    cleaned_text = " ".join(words)
    return cleaned_text


# Apply the preprocess_text function to the 'SentimentText' column and create a new 'cleaned_text' column
df["cleaned_text"] = df["SentimentText"].head(50).apply(preprocess_text)

# Display the DataFrame with the original and cleaned text
print(df[["SentimentText", "cleaned_text"]].head(50))
print()

# fill missing values with empty string
df["cleaned_text"].fillna("", inplace=True)

# generate TF-IDF for 200 features and print out vocabulary
tfidf_vectorizer = TfidfVectorizer(max_features=200)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])
vocabulary = tfidf_vectorizer.get_feature_names_out()

# print vocabulary for before cleanup
print(f"Vocabulary before cleanup: \n{vocabulary}\n")

# print vocabulary after cleanup
print(f"Vocabulary after cleanup: \n{tfidf_matrix.toarray()}\n")

# print out the dimensions of the matrix
print(f"Matrix Dimensions: {tfidf_matrix.shape}")


