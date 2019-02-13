import numpy as numpy
import torch
from torch import nn

class EncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        self.lstm = nn.LSTM(feature_size, self.hidden_size)

    def forward(self, input, past_state=None):
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = ( torch.randn((1,input.shape[1],self.hidden_size)) , torch.randn((1,input.shape[1],self.hidden_size)) )
        encodings, state = self.lstm(input, past_state)
        return encodings, state


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, past_state, out_len):
        batch_size = past_state[0].shape[1]
        hidden_state = past_state[0].view((batch_size,-1))
        cell_state = past_state[1].view((batch_size,-1))
        lstm_in = torch.zeros((batch_size,self.output_size))
        output = torch.zeros((out_len, batch_size, self.output_size))
        for i in range(out_len,0,-1):
            hidden_state, cell_state = self.lstm(lstm_in, (hidden_state, cell_state))
            lstm_in = self.softmax(self.out(hidden_state))
            output[i-1,:,:] = lstm_in
        return output



class RiskPredictor(nn.Module):
    def __init__(self, encoding_size, demographic_size):
        super(RiskPredictor, self).__init__()
        self.encoding_size = encoding_size 
        self.demographic_size = demographic_size
        self.net = nn.Sequential( nn.Linear(self.encoding_size+self.demographic_size, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 1) )


    def forward(self, encoding, demographics):
        x = torch.cat((encoding,demographics), dim=1) 
        risk = nn.Sigmoid()(self.net(x))  
        return risk
