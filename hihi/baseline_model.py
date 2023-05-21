import torch
from torch import nn
from transformers import AutoTokenizer

class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.checkpoint = "beomi/KcELECTRA-base-v2022"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.embedding = nn.Embedding(num_embeddings=54329, embedding_dim=input_size)
        self.dropout = 0
        self.num_layers = 1
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers,
                          nonlinearity="tanh", batch_first=True).to(self.device)
        self.linear = nn.Sequential(nn.Linear(hidden_size, 10),nn.Linear(10, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor, return_embedding=False):
        embeddings = self.embedding(X)
        N = X.shape[0]
        h_0 = torch.zeros((self.num_layers, N, self.hidden_size)).to(self.device)
        y, _ = self.rnn(embeddings, h_0)
        output = self.sigmoid(self.linear(y[:, -1, :]))
        if return_embedding:
            return output, y[:, -1, :]
        else:
            return output

    def tokenize_texts(self, texts):
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors="pt")
        return tokenized_texts['input_ids'].to(self.device)