import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.checkpoint = "beomi/KcELECTRA-base-v2022"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.pretrained_model = AutoModel.from_pretrained(self.checkpoint).to(self.device)
        self.dropout = 0
        self.num_layers = 1
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers,
                          nonlinearity="tanh", batch_first=True).to(self.device)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor, return_embedding=False):
        N = X.shape[0]
        h_0 = torch.zeros((self.num_layers, N, self.hidden_size)).to(self.device)
        y, _ = self.rnn(X, h_0)
        output = self.sigmoid(self.linear(y[:, -1, :]))
        if return_embedding:
            return output, y[:, -1, :]
        else:
            return output

    def embed_texts(self, texts: list):
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors="pt")
        model_output = self.pretrained_model(input_ids=tokenized_texts['input_ids'].to(self.device),
                                             attention_mask=tokenized_texts['attention_mask'].to(self.device))
        embeddings = model_output["last_hidden_state"]
        return embeddings
