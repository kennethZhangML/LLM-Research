import torch 
import torch.nn as nn 
import torch.functional as F 

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X):
        embedded = self.dropout(self.embedding(X))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell 
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W_a = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(seq_len, 1, 1)

        energy = torch.tanh(self.W_a(torch.cat((hidden, encoder_outputs), dim = 2)))
        attention = self.v_a(energy).squeeze(2)
        return F.softmax(attention, dim = 0)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()

        self.output_dim = output_dim 
        self.attention = attention 

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, n_layers, dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, hidden, cell, encoder_outputs):
        inputs = inputs.unsqueeze(0)
        embedded = self.dropout(self.embedding(inputs))

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        weighted = torch.bmm(attn_weights, encoder_outputs.permute(1, 0, 2))
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell 
