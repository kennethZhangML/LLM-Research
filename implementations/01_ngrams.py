import numpy as np 
import nltk 
from collections import defaultdict, Counter 
from nltk.util import ngrams 
from ntlk.corpus import reuters

nltk.download('reuters')
nltk.download('punkt')

class NGramModel:
    def __init__(self, n):
        self.n = n 
        self.ngrams = defaultdict(int)
    
    def train(self, tokens):
        for i in range(len(tokens) - self.n + 1):
            self.ngrams[tuple(tokens[i : i + self.n])] += 1
    
    def predict(self, context):
        context = tuple(context)
        predictions = {}
        for ngram, count in self.ngrams.items():
            if ngram[:-1] == context:
                predictions[ngram[-1]] = count 
        return predictions 

class NGramLaPlace:
    def __init__(self, n):
        self.n = n 
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
    
    def train(self, sentences):
        for sentence in sentences:
            sentence = ["<s>"] * (self.n - 1) + sentence + ["<s>"]

            for i in range(len(sentence) - self.n + 1):
                ngram_tuple = tuple(sentence[i : i + self.n])
                self.ngram_counts[ngram_tuple] += 1
                self.context_counts[tuple(sentence[i : i + self.n - 1])] += 1

    def generate_text(self, context, num_words):
        text = context 
        for _ in range(num_words):
            context_ngram = tuple(text[-(self.n - 1): ])
            next_word = self._predict(context_ngram)
            text.append(next_word)
        return text 
    
    def _predict(self, context_ngram):
        max_prob = 0
        best_next_word = "</s>"

        for word in self.ngram_counts.keys():
            prob = self._probability(word[-1], context_ngram)
            if prob > max_prob:
                max_prob = prob 
                best_next_word = word[-1]
        return best_next_word
    
    def _probability(self, word, context_ngram):
        ngram = context_ngram + (word, )
        return (self.n_gram_counts[ngram] + 1) / (self.context_counts[context_ngram] + len(self.ngram_counts))

if __name__ == "__main__":
    sentences = reuters.sents()
    model = NGramLaPlace(3)
    model.train(sentences)

    context = ["world", "is"]
    print(model.generate_text(context, 10))
    


