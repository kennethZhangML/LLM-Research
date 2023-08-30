import torch 
import torch.nn as nn 

# For this example, we will use the pre-made and pre-trained BERT models and load in there weights
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

sentence = "The cat sat on the [MASK]."

tokens = tokenizer.tokenize(sentence)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
masked_index = tokens.index("[MASK]")
indexed_tokens[masked_index] = tokenizer.mask_token_id 

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensor = torch.tensor([[0] * len(indexed_tokens)])

with torch.no_grad():
    predictions = bert_mlm(tokens_tensor, segments_tensor)[0]


