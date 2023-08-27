import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads 
        self.d_model = d_model 

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask = None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_model // self.num_heads).transepose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(self.d_model // self.num_heads)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim = -1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(output)
        return output 

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadedAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attnd = self.attention(x, x, x, mask)
        x = self.norm1(attnd + x)
        ff_out = self.feed_forward(x)
        return self.norm2(ff_out + x)

class BidirectionalEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout = 0.1):
        super(BidirectionalEncoder, self).__init__()

        self.enc_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, x, mask = None):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x