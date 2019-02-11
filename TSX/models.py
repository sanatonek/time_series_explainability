import numpy as numpy
import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, feature_size=1, hidden_size=10):
        super(RNN, self).__init__()
        self.rnn = nn.LSTMCell(input_size=feature_size, hidden_size=hidden_size)
        self.risk = nn.Linear(hidden_size,1)

    def forward(self, x, past):
        h_out, c_out = self.rnn(x, past)
        risk = nn.Sigmoid()(self.risk(h_out))
        return risk, h_out, c_out