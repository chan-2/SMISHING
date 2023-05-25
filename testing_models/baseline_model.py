import torch
from torch import nn


class BaselineModel(nn.Module):

    def forward(self, X: torch.Tensor, return_embedding=False):
        embeddings = self.embedding(X)
        N = X.shape[0]
        h_0 = torch.zeros((self.num_layers, N, self.hidden_size))
        y, _ = self.rnn(embeddings, h_0)
        output = self.sigmoid(self.linear(y[:, -1, :]))
        if return_embedding:
            return output, y[:, -1, :]
        else:
            return output

    def tokenize_texts(self, texts):
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors="pt")
        return tokenized_texts['input_ids']