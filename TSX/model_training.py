import torch
from torch import nn
from models import EncoderRNN, DecoderRNN, Encoder, Decoder
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np


def train_ecoder(encoder_model, decoder_model,x, n_epochs=100):
    ''' Function to train the time series autoencoder
    Args:
        x: Time series input of shape (seq_len, batch, feature_size)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    x = x.to(device)
    encoder_model.train()
    decoder_model.train()
    seq_len = x.shape[0]

    parameters = list(encoder_model.parameters())+list(decoder_model.parameters())
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(parameters)

    for i in range(n_epochs):
        optimizer.zero_grad()
        encodings = encoder_model(x)
        x_recon = decoder_model(encodings)
        # encodings, state = encoder_model(x)
        # x_recon = decoder_model(state,seq_len)
        loss = criterion(x_recon,x)
        loss.backward()
        optimizer.step()
        if i%10==0:
            print('Training loss: ', loss.item())
    return x_recon

def test_ecoder(encoder_model, decoder_model,x, n_epochs=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    x = x.to(device)
    encoder_model.eval()
    decoder_model.eval()

    criterion = nn.MSELoss()
    encodings = encoder_model(x)
    x_recon = decoder_model(encodings)
    loss = criterion(x_recon,x)
    print('Test loss: ', loss.item())
    return x_recon

if __name__=='__main__':
    # signals = [[float(x.split('\t')[0]), float(x.split('\t')[1]), float(x.split('\t')[2]), float(x.split('\t')[3]), float(x.split('\t')[4])]
    #              for x in open("./test_data/mHealth_subject1.log").readlines()]
    # signals = np.array(signals)
    # print(signals.shape)
    # x = []
    # for i in range(100):
    #     x.append(signals[i:i+50,:])
    # x = np.array(x).transpose((1,0,2))
    np.random.seed(2019)
    Fs = 50
    f = 5
    sample = 48
    x=[]
    for i in range(100):
        Fs = 100+10*i
        t = np.arange(sample)
        y = np.sin(2 * np.pi * f * t / Fs)
        x.append(y)
    x = (np.array(x)).reshape(-1,sample,1)
    x = x.transpose((0,2,1))
    # x = torch.randn((100,10,5))
    # enc = EncoderRNN(feature_size=1, hidden_size=5)
    # dec = DecoderRNN(hidden_size=5, output_size=1)
    # enc = Encoder(50,5)
    dec = Decoder(48,64)
    # input shape to the encoder = [batch,in_channel,input_len]
    enc = Encoder(1)
    encoding = (enc(torch.Tensor(x))).detach().numpy()
    # plt.plot(encoding[0,:])
    # plt.show()

    x_recon = train_ecoder(encoder_model=enc, decoder_model=dec,x=torch.Tensor(x), n_epochs=500)
    x_recon = x_recon.detach().numpy()

    plt.plot(np.array(x_recon[3,:,:]).reshape(-1,))
    plt.plot(np.array(x[3,:,:]).reshape(-1,))
    plt.title('Training reconstruction')
    plt.show()


    Fs = 10
    f = 5
    sample = 48
    x_test=[]
    for i in range(10):
        Fs = 100+10*i
        t = np.arange(sample)
        y = np.sin(2 * np.pi * f * t / Fs)
        x_test.append(y)
    x_test = (np.array(x_test)).reshape(-1,sample,1)
    x_test = x_test.transpose((0,2,1))

    x_recon_test = test_ecoder(encoder_model=enc, decoder_model=dec,x=torch.Tensor(x_test))
    x_recon_test = x_recon_test.detach().numpy()

    plt.figure()
    plt.plot(np.array(x_recon_test[3,:,:]).reshape(-1,))
    plt.plot(np.array(x_test[3,:,:]).reshape(-1,))
    plt.title('Test reconstruction')
    plt.show()
