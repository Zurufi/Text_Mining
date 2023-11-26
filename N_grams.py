# Ali Alzurufi
# Professor Lauren
# Date: October 15 2023
# MCS 5223: Text Mining and Data Analytics

from collections import defaultdict, Counter

class BigramModel:
    def __init__(self):
        self.model = defaultdict(Counter)

    def train(self, text):
        tokens = text.split()
        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i + 1]
            self.model[current_word][next_word] += 1

        # Convert counts to probabilities
        for current_word, next_words in self.model.items():
            total_count = float(sum(next_words.values()))
            for next_word, count in next_words.items():
                self.model[current_word][next_word] = count / total_count

    def predict_next_word(self, current_word):
        return self.model[current_word]

    def generate_sentence(self, start_word, max_length=10):
        current_word = start_word
        sentence = [current_word]

        for i in range(max_length - 1):
            next_words_probs = self.predict_next_word(current_word)
            if not next_words_probs:
                break
            next_word = max(next_words_probs, key=next_words_probs.get)
            sentence.append(next_word)
            current_word = next_word

        return ' '.join(sentence)

# Example usage with toy data
toy_data = """
The cat sat on the mat.
The dog barked at the cat.
The cat meowed and ran away.
The dog chased the cat.
The cat climbed the tree.
Dogs and cats can be friends.
"""

model = BigramModel()
model.train(toy_data)

print(model.generate_sentence("on"))


# In[2]:


from collections import defaultdict, Counter
import random

class TrigramModel:
    def __init__(self):
        self.model = defaultdict(Counter)

        #  loop through and create tuple for current word and next word
    def train(self, text):
        tokens = text.split()
        for i in range(len(tokens) - 2):
            current_word = tokens[i]
            next_word = tokens[i + 1]
            next_next_word = tokens[i + 2]
            self.model[(current_word, next_word)][next_next_word] += 1

        # Convert counts to probabilities
        for word_pair, next_words in self.model.items():
            total_count = float(sum(next_words.values()))
            for next_word, count in next_words.items():
                self.model[word_pair][next_word] = count / total_count

    def predict_next_word(self, word_pair):
        return self.model[word_pair]

    def generate_sentence(self, start_words, max_length=10):
        current_words = start_words
        sentence = list(current_words)

        for i in range(max_length - 2):
            next_words_probs = self.predict_next_word(current_words)
            if not next_words_probs:
                break
            next_word = max(next_words_probs, key=next_words_probs.get)
            sentence.append(next_word)
            current_words = (current_words[1], next_word)

        return ' '.join(sentence)

# Example usage with toy data
toy_data = """
The cat sat on the mat.
The dog barked at the cat.
The cat meowed and ran away.
The dog chased the cat.
The cat climbed the tree.
Dogs and cats can be friends.
"""

model = TrigramModel()
model.train(toy_data)

print(model.generate_sentence(("climbed", "the")))  



