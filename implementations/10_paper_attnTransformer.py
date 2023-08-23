import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.keys = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)
    
    # Returns the output and attention layer
    # softmax(matmul(q, k) / scale by d_k ) 
    def attention(self, q, k, v, mask = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim = -1)
        output = torch.matmul(attn, v)
        return output, attn 

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.query(k).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.query(v).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        x, _ = self.attention(q, k, v, mask)

        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.fc(x)

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, X):
        return self.fc2(F.relu(self.fc1(X)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask = None):
        attn_output = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x
    
if __name__ == "__main__":
    model = TransformerBlock(d_model = 512, num_heads = 8, d_ff = 2048)
    x = torch.randn(16, 10, 512)
    output = model(x)

    print(output)