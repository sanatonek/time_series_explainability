import numpy as numpy
import torch
from torch import nn
import torch.nn.functional as F

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


class Encoder(nn.Module):
    ''' This encoder consists of a CNN and a RNN layer on top of it, 
        in order to be ableto handle variable length inputs
    '''
    def __init__(self, in_channels, encoding_size=64):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.enc_channels = 32
        self.encoding_size = encoding_size
        self.enc_cnn = nn.Sequential( nn.Conv1d(self.in_channels, self.enc_channels, kernel_size=4),
                                      nn.MaxPool1d(kernel_size=2, stride=2),
                                      nn.ReLU()  )
        self.enc_lstm = nn.LSTMCell(input_size=self.enc_channels, hidden_size=self.encoding_size)

    def forward(self, signals):
        code = self.enc_cnn(signals)
        h = torch.zeros(code.shape[0], self.encoding_size)
        c = torch.zeros(code.shape[0], self.encoding_size)
        for i in range(code.shape[2]):
            h, c = self.enc_lstm(code[:,:,i].view(code.shape[0],code.shape[1]) , (h,c))
        return nn.Softmax(dim=1)(h)



class Decoder(nn.Module):
    def __init__(self, output_len, code_size):
        super(Decoder, self).__init__()
        self.code_size = code_size
        self.output_len = output_len
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, self.output_len)


    def forward(self, code):
        out = F.relu(self.dec_linear_1(code))
        out = self.dec_linear_2(out)
        out = out.view([code.size(0), 1,self.output_len])
        return out