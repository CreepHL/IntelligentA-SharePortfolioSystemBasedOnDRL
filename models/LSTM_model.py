import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=0.01)

    def forward(self, input_sequence):
        batch_size, seq_len = input_sequence.shape[0], input_sequence.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_sequence, (h_0, c_0))
        return output, h


class LSTMMain(nn.Module):
    # def __init__(self, input_size, output_len, lstm_hidden, lstm_layers, batch_size, device="cpu"):
    #     super(LSTMMain, self).__init__()
    #     self.lstm_hidden = lstm_hidden
    #     self.lstm_layers = lstm_layers
    #     self.lstmunit = LSTMModel(input_size, lstm_hidden, lstm_layers, batch_size, device)
    #     self.linear = nn.Linear(lstm_hidden, output_len)
    #
    # def forward(self, input_seq):
    #     ula, h_out = self.lstmunit(input_seq)
    #     out = ula.contiguous().view(ula.shape[0] * ula.shape[1], self.lstm_hidden)
    #     out = self.linear(out)
    #     out = out.view(ula.shape[0], ula.shape[1], -1)
    #     out = out[:, -1, :]
    #     return out

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2,  device="cpu"):
        super(LSTMMain, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
