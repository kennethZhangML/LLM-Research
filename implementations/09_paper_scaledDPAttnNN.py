import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ScaledDPAttn(nn.Module):
    def __init__(self, dropout = 0.1):
        super(ScaledDPAttn, self).__init__()

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask = None):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, dim = -1)
        
        attn_weights = F.softmax(scores, dim = -1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)
        return output, attn_weights
    
if __name__ == "__main__":
    attn_layer = ScaledDPAttn()
    query = torch.randn(32, 10, 64)
    key = torch.randn(32, 10, 64)
    value = torch.randn(32, 10, 64)

    output, attn_weights = attn_layer(query, key, value)
    print(output.shape)
    print(attn_weights)