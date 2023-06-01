import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.functional import F

class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.checkpoint = "beomi/KcELECTRA-base-v2022"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.pretrained_model = AutoModel.from_pretrained(self.checkpoint).to(self.device)
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, device=self.device)
        self.att_linear = nn.Linear(hidden_size, hidden_size)
        self.cvt_linear = nn.Linear(2*hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor, return_embedding=False):
        N = X.shape[0]
        L = X.shape[1]
        h = torch.zeros((N, self.hidden_size)).to(self.device)
        c = torch.zeros((N, self.hidden_size)).to(self.device)
        hs = h.view(N, 1, self.hidden_size)
        h_tilde = torch.zeros((N, self.hidden_size)).to(self.device)
        for i in range(L):
            h, c = self.lstm_cell(X[:,i,:], (h_tilde, c))
            attention_weights = self._get_attention_scores(hs, h, N, softmax=True)
            h_hat = torch.sum(hs * attention_weights, dim=1)
            h_tilde = self.cvt_linear(torch.cat((h, h_hat), dim=1))
            hs = torch.cat((hs, h_tilde.view(N, 1, self.hidden_size)), dim=1)

        output = self.sigmoid(self.out_linear(h))
        if return_embedding:
            return output, h
        else:
            return output
    def _get_attention_scores(self, v1, v2, N, softmax=True):
        attention_scores = torch.bmm(v1.view(N, -1, self.hidden_size), self.att_linear(v2).view(N, self.hidden_size, 1))
        if softmax:
            return F.softmax(attention_scores, dim=1)
        else:
            return attention_scores
    def embed_texts(self, texts):
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors="pt")
        model_output = self.pretrained_model(input_ids=tokenized_texts['input_ids'].to(self.device),
                                             attention_mask=tokenized_texts['attention_mask'].to(self.device))
        embeddings = model_output["last_hidden_state"]
        return embeddings
