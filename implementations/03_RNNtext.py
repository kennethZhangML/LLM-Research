import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class RNNtext(nn.Module):
    def __init__(self, n_hidden, n_class):
        super(RNNtext, self).__init__()
        self.rnn = nn.RNN(n_class, n_hidden)
        
        self.W = nn.Linear(n_hidden, n_class, bias = False)
        self.b = nn.Parameter(torch.ones([n_class]))
    
    def forward(self, hidden, X):
        X = X.transpose(0, 1)
        outputs, hidden = self.rnn(X, hidden)
        outputs = outputs[-1]
        logits = self.W(outputs) + self.b 
        return logits 

def make_batches(sentences, word_dict, n_class):
    input_batches = []
    target_batches = []

    for sent in sentences:
        words = sent.split()
        inputs = [word_dict[n] for n in words[:-1]]
        target = [word_dict[-1]]

        input_batches.append(np.eye(n_class)[inputs])
        target_batches.append(target)
    
    return input_batches, target_batches 

if __name__ == "__main__":
    n_step = 2 
    n_hidden = 5

    sentences = ["i like dogs", "i like coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))

    word_dict = {w: i for i, w in enumerate(word_list)}
    num_dict = {i: w for i, w in enumerate(word_list)}

    n_class = len(word_dict)
    batch_size = len(sentences)

    model = RNNtext(n_hidden, n_class)
    criterion = nn.CrossEntropyLoss().cuda() 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    input_batch, target_batch = make_batches(sentences, word_dict, n_class)
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in range(5000):
        optimizer.step()

        hidden_init = torch.zeors(1, batch_size, n_hidden)
        output = model(hidden_init, input_batch)

        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
        
        loss.backward()
        optimizer.step()
    
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim = True)[1]
    print([sent.split()[:2] for sent in sentences], '->', [num_dict[n.item()] for n in predict.squeeze()])