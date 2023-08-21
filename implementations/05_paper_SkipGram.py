import numpy as np 

import torch 
import torch.nn as nn 

import matplotlib.pyplot as plt

# The following code is an implementation for the paper:
# Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). 
# Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26.

# Includes:
# 1. SkipGram model an Embedding and Linear Layer
# 2. Vocab creation using sentence array to word dictionary
# 3. skip gram creation following the algorithm from the paper
# 4. Training loop 

sentences = [
    "The cat sat on the mat",
    "Dog and cat are friends",
    "We have a mat in our house"
]

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        embeds = self.embeddings(x)
        out = self.linear(embeds)
        log_probs = nn.functional.log_softmax(out, dim = 1)
        return log_probs

def create_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        for word in sentence.split():
            vocab.add(word.lower())
    
    vocab = list(vocab)
    vocab_size = len(vocab)
    word_index = {w: i for i, w in enumerate(vocab)}
    return vocab, vocab_size, word_index 

def skip_grams(sentences, word_dict):
    pair_targets = []

    for sentence in sentences:
        tokens = [word_dict[word.lower()] for word in sentence.split()]
        for i, token in enumerate(tokens):
            for j in range(-2, 3):
                context_pos = i + j
                if j != 0 and 0 <= context_pos < len(tokens):
                    pair_targets.append((token, tokens[context_pos]))
    return pair_targets


if __name__ == "__main__":
    vocab, vocab_size, word_dict = create_vocab(sentences)
    pair_grams = skip_grams(sentences, word_dict)

    model = SkipGram(vocab_size, 100)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    epochs = 100 
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for (input_word, target_word) in pair_grams:
            optimizer.zero_grad()

            input_vector = torch.LongTensor([input_word])
            log_probs = model(input_vector)

            loss = criterion(log_probs, torch.LongTensor([target_word]))
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            losses.append(total_loss / len(pair_grams))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(pair_grams)}')
        
    embeddings = model.embeddings.weight.data 
    print(embeddings[word_dict['cat']])


