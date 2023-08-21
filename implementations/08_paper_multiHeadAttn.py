import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads 
        self.d_model = d_model 
        assert d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads 
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        return x
    
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attn_logits = torch.matmul(query, key.transpose(-2, -1)) / self.depth ** 0.5

        if mask is not None:
            scaled_attn_logits += (mask * -1e9)
        
        attn_weights = F.softmax(scaled_attn_logits, dim = -1)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(output)

if __name__ == "__main__":
    mha = MultiHeadAttention(d_model = 512, num_heads = 8)
    x = torch.randn(32, 10, 512)
    output = mha(x, x, x)
    print(output.shape)