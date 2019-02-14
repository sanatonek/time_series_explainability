import torch
from torch import nn
from models import EncoderRNN, DecoderRNN


def train_ecoder(encoder_model, decoder_model,x, n_epochs=100):
    ''' Function to train the time series autoencoder
    Args:
        x: Time series input of shape (seq_len, batch, feature_size)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    x = x.to(device)
    seq_len = x.shape[0]

    parameters = list(encoder_model.parameters())+list(decoder_model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters)

    for i in range(n_epochs):
        encodings, state = encoder_model(x)
        x_recon = decoder_model(state,seq_len)
        loss = criterion(x_recon,x)
        loss.backward()
        optimizer.step()
        if i%10==0:
            print('Training loss: ', loss.item())



if __name__=='__main__':
    x = torch.randn((100,10,5))
    enc = EncoderRNN(feature_size=5, hidden_size=100)
    dec = DecoderRNN(hidden_size=100, output_size=5)
    train_ecoder(encoder_model=enc, decoder_model=dec,x=x, n_epochs=500)
