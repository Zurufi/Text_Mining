# Ali Alzurufi
# Professor Lauren
# Date: October 15 2023
# MCS 5223: Text Mining and Data Analytics

from bs4 import BeautifulSoup
import requests
import re


def transformURLtoString(url):
    result = requests.get(url)
    html = result.text
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

text = transformURLtoString('https://arstechnica.com/science/2021/09/ny-prepared-for-tens-of-thousands-of-unvaccinated-health-workers-to-lose-jobs/')


import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter


# 1. Text Cleaning
cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

# 2. Tokenization
tokens = word_tokenize(cleaned_text)

# 3. Stop Word Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.lower() not in stop_words]

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

# 5. Named Entity Recognition
nlp = spacy.load("en_core_web_sm")
text = " ".join(lemmatized_words)
doc = nlp(text)

# modify text with underscores for multi-word entities
def modify_entity(text, entities):
    for ent in entities:
        if " " in ent.text:
            text = text.replace(ent.text, ent.text.replace(" ", "_"))
    return text

# Extract entities with underscore
entities = [(modify_entity(ent.text, doc.ents), ent.label_) for ent in doc.ents]

# Results
print("1. Text Cleaning:")
print(cleaned_text)

print("\n2. Tokenization:")
print(tokens)

print("\n3. Stop Word Removal:")
print(filtered_words)

print("\n4. Lemmatization:")
print(lemmatized_words)

print("\n5. Named Entity Recognition (NER):")
print(entities)

# Count and display entity labels
labels = [ent.label_ for ent in doc.ents]
Counter(labels)



import spacy

# load sm model
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

collocation_replacements = {}

for ent in doc.ents:
    collocation_replacements[ent.text] = ent.text.replace(" ", "_")

def replace_collocations(text):
    for collocation, replacement in collocation_replacements.items():
        text = text.replace(collocation, replacement)
    return text

# Replace collocations in the original text
updated_text = replace_collocations(text)

print(updated_text)


from spacy import displacy

displacy.render(doc, jupyter=True, style='ent')


