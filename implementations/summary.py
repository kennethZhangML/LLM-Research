import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import defaultdict, Counter
from gensim.models import Word2Vec

# ---------------------- NGram Model ---------------------- #
class NGram:
    """
    The N-gram model predicts the next word in a sequence based on the preceding 'n-1' words.
    """
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(int)

    def train(self, tokens):
        for i in range(len(tokens) - self.n + 1):
            self.ngrams[tuple(tokens[i:i+self.n])] += 1
  
    def predict(self, context):
        context = tuple(context)
        predictions = {}
        for ngram, count in self.ngrams.items():
            if ngram[:-1] == context:
                predictions[ngram[-1]] = count
        return predictions


# ---------------------- Word2Vec & SkipGram Model ---------------------- #
class Word2VecModel(nn.Module):
    """
    Word2Vec with the SkipGram approach.
    It predicts context words (surrounding words) from a target word (center word).
    The main idea is to maximize the probability of predicting surrounding words given a center word.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Word2VecModel, self).__init__()

        self.W = nn.Linear(input_dim, hidden_dim)
        self.WT = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        x = self.W(X)
        x = self.WT(x)
        return x 


# ---------------------- RNN Model ---------------------- #
class SimpleRNN(nn.Module):
    """
    Recurrent Neural Networks (RNNs) remember past information through internal memory.
    They are useful for sequence-to-sequence tasks.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


# ---------------------- LSTM Model ---------------------- #
class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) is an improvement over RNNs,
    designed to remember long-term dependencies using gates.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ---------------------- CNN Model ---------------------- #
class CNNModel(nn.Module):
    """
    Convolutional Neural Networks (CNNs) use convolutional layers to filter input data for useful information.
    They're primarily used for image recognition but can be used for sequence data in NLP.
    """
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 28 * 28, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
