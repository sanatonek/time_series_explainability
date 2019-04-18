from TSX.utils import load_data
import random
import torch
import os
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal


class Generator(torch.nn.Module):
    def __init__(self, feature_size, seed=random.seed('2019')):
        super(Generator, self).__init__()
        self.hidden_size = 100
        self.seed = seed
        self.feature_size = feature_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rnn = torch.nn.GRU(self.feature_size, self.hidden_size)
        self.regressor = torch.nn.Sequential(torch.nn.BatchNorm1d(num_features=self.hidden_size),
                                             #torch.nn.Dropout(0.5),
                                             torch.nn.Linear(self.hidden_size, self.feature_size),
                                             torch.nn.Sigmoid())

    def forward(self, x, past_state=None):
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        x = x.permute(2, 0, 1)
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, x.shape[1], self.hidden_size]).to(self.device)
        all_encoding, encoding = self.rnn(x, past_state)
        mus = self.regressor(encoding.view(encoding.shape[1], -1))
        #p_xs = [MultivariateNormal(mu, torch.eye(self.feature_size).to(self.device))for mu in mus]
        reparam_samples = mus + torch.randn_like(mus).to(self.device)*0.1
        return reparam_samples, mus


def train_generator(generator_model, train_loader, valid_loader):
    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator_model.to(device)

    parameters = generator_model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.0001, weight_decay=1e-3)
    loss_criterion = torch.nn.MSELoss()

    for epoch in range(30 + 1):
        generator_model.train()
        epoch_loss = 0
        for i, (signals, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            signals = signals[:,:-4,:]
            #t = random.randint(12, signals.shape[2]-1)
            t=12
            signals = torch.Tensor(signals.float()).to(device)
            next_timestep, mus = generator_model(signals[:,:,:t])
            reconstruction_loss = loss_criterion(next_timestep, signals[:, :, t])
            epoch_loss = + reconstruction_loss.item()
            reconstruction_loss.backward()
            optimizer.step()
        test_loss = test_generator(generator_model, valid_loader)

        train_loss_trend.append(epoch_loss)
        test_loss_trend.append(test_loss)
        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', epoch_loss)
            print('Test ===>loss: ', test_loss)
    # Save model and results
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    torch.save(generator_model.state_dict(), './ckpt/generator.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig('generator_train_loss.png')


def test_generator(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    for i, (signals, labels) in enumerate(test_loader):
        signals = signals[:, :-4, :]
        t = random.randint(12, signals.shape[2] - 1)
        #t=12
        signals = torch.Tensor(signals.float()).to(device)
        next_timestep, mus = model(signals[:,:,:t])
        loss = torch.nn.MSELoss()(next_timestep, signals[:,:,t])
        test_loss = + loss.item()
    return test_loss


#batch_size = 100
#p_data, train_loader, valid_loader, test_loader = load_data(batch_size, './data_generator/data')
#generator = Generator(p_data.feature_size-4)
#train_generator(generator, train_loader, valid_loader)
