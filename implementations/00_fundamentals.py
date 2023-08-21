import numpy as np 
import spacy 
import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Splitting text into smaller chunks called tokens 
def tokenize(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]
    print(tokens)

# Reduce words to their base/roots form
def stemmer(word):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos = "v")
    return stem, lemma 

# Labelling words with respective parts of speech 
def partOFspeech(text):
    return nltk.pos_tag(text)

# identify entities like names, places, dates 
def namedEntityRecog(text):
    entities = []
    labels = []

    for ent in text.ents:
        print(ent.text, ent.label_)
        entities.append(ent.text)
        labels.append(ent.label_)
    return entities, labels

# Understanding the grammatical structure of sentences 
# dependency and constituency parsing 
def parsing(doc):
    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_, 
              chunk.root.head.text)
    







