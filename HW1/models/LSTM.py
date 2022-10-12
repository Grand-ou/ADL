import torch
import torch.autograd as autograd
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, output_size, sequence_len=10, input_feature_dim=26, hidden_dim=128, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.embedding = nn.Embedding(sequence_len, input_feature_dim)
        self.lstm = nn.LSTM(input_feature_dim, hidden_dim, num_layers, batch_first=False, dropout=0.3)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout_layer = nn.Dropout(p=0.5)

    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch):
        # self.hidden = self.init_hidden(batch.size(-2))
        batch = batch.permute(1, 0, 2)
        outputs, (ht, ct) = self.lstm(batch)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)
        return output