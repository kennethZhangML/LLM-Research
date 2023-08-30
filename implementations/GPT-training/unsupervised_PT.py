import torch 
import torch.nn as nn

voc_size = 10000
embedding_dim = 256
context_window = 5
num_layers = 4
hidden_size = 512

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerBlock, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + attn_output 
        x = self.layer_norm1(x)

        ff_out = self.ff(x)
        x = x + ff_out 
        x = self.layer_norm2(x)
        return x 

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_window, num_layers):
        super(LanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_window, embedding_dim)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads = 8) for _ in range(num_layers)]
        )

    def forward(self, input_sequence):
        context_embeddings = self.token_embedding(input_sequence)
        batch_size, seq_len, _ = context_embeddings.size()
        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_sequence.device)
        position_embeddings = self.position_embedding(position_ids)

        context_embeddings += position_embeddings

        for transformer_block in self.transformer_blocks:
            context_embeddings = transformer_block(context_embeddings)
        return context_embeddings

if __name__ == "__main__":
    batch_size = 32 
    seq_len = 20 
    input_data = torch.randint(0, voc_size, (batch_size, seq_len))

    model = LanguageModel(voc_size, embedding_dim, context_window, num_layers)
    output = model(input_data)
    print(output.shape)



