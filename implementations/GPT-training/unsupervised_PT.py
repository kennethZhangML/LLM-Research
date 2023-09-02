import torch 
import torch.nn as nn 

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output 
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + ff_output 
        x = self.norm2(x)
        return x 
    
class UnsupervisedPTModel(nn.Module):
    def __init__(self, voc_size, d_model, num_head, dim_ff, num_layers, context_size):
        super(UnsupervisedPTModel, self).__init__()
        self.token_embedding = nn.Embedding(voc_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_head, dim_ff)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, voc_size)

    def forward(self, context):
        token_embedded = self.token_embedding(context)
        position_ids = torch.arange(context.size(1), device = context.device).unsqueeze(0)
        position_embedded = self.position_embedding(position_ids)
        x = token_embedded + position_embedded
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        output = self.output_layer(x)
        return output
    
if __name__ == "__main__":
    batch_size = 32
    voc_size = 10000
    d_model = 512 
    num_head = 8
    dim_ff = 2048 
    num_layers = 6
    context_size = 10 

    model = UnsupervisedPTModel(voc_size, d_model, num_head, dim_ff, num_layers, context_size)
    input_data = torch.randint(0, voc_size, (batch_size, context_size))
    output = model(input_data)

    print("Output: ", output)
