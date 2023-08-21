import numpy as np 

import torch 
import torch.nn as nn 

from torch.utils.data import Dataset, DataLoader

class LSTMtext(nn.Module):
    def __init__(self, n_class, n_hidden):
        self.n_hidden = n_hidden
        self.n_class = n_class 

        super(LSTMtext, self).__init__()
        self.lstm = nn.LSTM(n_class, n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias = False)
        self.b = nn.Parameter(torch.ones([self.n_class]))

    def forward(self, X):
        x = X.transpose(0, 1)
        hidden_state = torch.zeros(1, len(X), self.n_hidden)
        cell_state = torch.zeros(1, len(X), self.n_hidden)

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]
        model = self.W(outputs) + self.b 
        return model 
    
class VariableLenDataset(Dataset):
    def __init__(self, seq_data, word_dict):
        self.seq_data = seq_data 
        self.word_dict = word_dict
    
    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        seq = self.seq_data[idx]
        inputs = [self.word_dict[n] for n in self.seq_data[:-1]]
        target = self.word_dict[seq[-1]]
        return inputs, target 
    
def create_batches(sentences, word_dict, n_class):
    input_batch, target_batch = [], []

    for sent in sentences:
        inputs = [word_dict[n] for n in sent[:-1]]
        target = word_dict[sent[-1]]
        input_batch.append(np.eye(n_class)[inputs])
        target_batch.append(target)
    return input_batch, target_batch 

if __name__ == "__main__":
    n_hidden = 128

    char_arr = [c for c in "SEPabcdefghijklmnopqrstuvwxyz"]
    word_dict = {n: i for i, n in enumerate(char_arr)}
    num_dict = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word_dict)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hashs', 'star']

    model = LSTMtext(n_class, n_hidden)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    dataset = VariableLenDataset(seq_data, word_dict)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, 
                            collate_fn = lambda x: zip(*x))
    
    for epoch in range(1000):
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            seq_lengths = [len(seq) for seq in inputs]
            max_seq_len = max(seq_lengths)

            input_batch = torch.zeros(max_seq_len, len(inputs), n_class)
            target_batch = torch.LongTensor(targets)

            for i, seq in enumerate(inputs):
                input_batch[:len(seq), i, :] = torch.LongTensor(seq)
            
            output = model(input_batch)
            loss = criterion(output, target_batch)

            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
        
    inputs = [sent[:3] for sent in seq_data]
    input_batch = torch.zeros(len(seq_data), max(len(seq) for seq in inputs))
    for i, seq in enumerate(inputs):
        input_batch[i, :len(seq), :] = torch.LongTensor([word_dict[char] for char in seq])
    
    predict = model(input_batch).data.max(1, keepdim = True)[1]
    print(inputs, '->', [num_dict[n.item()] for n in predict.squeeze()])


    

