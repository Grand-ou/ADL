from typing import Dict

import torch
from torch.nn import Embedding
from torch import nn

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        input_feature_dim: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(input_feature_dim, hidden_size, num_layers, batch_first=False, dropout=dropout, bidirectional = bidirectional)
        self.hidden2out = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = batch.permute(1, 0, 2)
        outputs, (ht, ct) = self.lstm(batch)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)
        return output


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
