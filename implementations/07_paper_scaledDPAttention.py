import torch
import torch.nn.functional as F 

def scaledDP_attention(query, key, value, mask = None):
    """ Scaled Dot-Product Attention with Masking"""

    d_k = query.size(1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim = -1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights    

if __name__ == "__main__":
    query = torch.randn(32, 10, 64)
    key = torch.randn(32, 10, 64)
    value = torch.randn(32, 10, 128)

    output, attention_weights = scaledDP_attention(query, key, value)
    print(output.shape)
    print(attention_weights)

    