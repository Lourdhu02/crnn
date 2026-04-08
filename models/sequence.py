import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True)

    def forward(self, x):
        out,_ = self.rnn(x)
        return out
