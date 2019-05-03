from TSX.utils import load_data, load_simulated_data
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


class FeatureGenerator(torch.nn.Module):
    def __init__(self, feature_size, hist=False, hidden_size=50, seed=random.seed('2019')):
        super(FeatureGenerator, self).__init__()
        self.seed = seed
        self.hist = hist
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.hist:
            self.rnn = torch.nn.GRU(self.feature_size, self.hidden_size)
            self.predictor = torch.nn.Sequential(torch.nn.Linear(self.feature_size-1+self.hidden_size, 20),
                                                 torch.nn.ReLU(),
                                                 torch.nn.BatchNorm1d(num_features=20),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(20, 1),
                                                 torch.nn.Sigmoid())
        else:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(self.feature_size-1, 20),
                                                 torch.nn.ReLU(),
                                                 torch.nn.BatchNorm1d(num_features=20),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(20, 1),
                                                 torch.nn.Sigmoid())

    def forward(self, x, past=None):
        if self.hist:
            past = past.permute(2, 0, 1)
            prev_state = torch.zeros([1, past.shape[1], self.hidden_size]).to(self.device)
            all_encoding, encoding = self.rnn(past.to(self.device), prev_state)
            x = torch.cat((encoding.view(encoding.size(1),-1), x), 1)
        mu = self.predictor(x)
        reparam_samples = mu + torch.randn_like(mu).to(self.device)*0.1
        return reparam_samples, mu


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


def train_feature_generator(generator_model, train_loader, valid_loader, feature_to_predict=1, n_epoch=30, historical=False):
    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator_model.to(device)

    parameters = generator_model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.0001, weight_decay=1e-3)
    loss_criterion = torch.nn.MSELoss()

    for epoch in range(n_epoch + 1):
        generator_model.train()
        epoch_loss = 0
        for i, (signals, _) in enumerate(train_loader):
            if historical:
                for t in [24]:#range(10,signals.shape[2]):
                    label = signals[:, feature_to_predict, t].contiguous().view(-1, 1)
                    signal = torch.cat((signals[:, :feature_to_predict, t], signals[:, feature_to_predict + 1:, t]), 1)

                    #print(signal.shape)
                    signal = signal.contiguous().view(-1, signals.shape[1] - 1)
                    signal = torch.Tensor(signal.float()).to(device)
                    past = signals[:,:,:t]
                    optimizer.zero_grad()
                    prediction, mus = generator_model(signal, past)
                    reconstruction_loss = loss_criterion(prediction, label.to(device))
                    epoch_loss = + reconstruction_loss.item()
                    reconstruction_loss.backward()
                    optimizer.step()

            else:
                original = signals[:,feature_to_predict,:].contiguous().view(-1,1)
                signal = torch.cat((signals[:,:feature_to_predict,:], signals[:,feature_to_predict+1:,:]), 1).permute(0,2,1)
                signal = signal.contiguous().view(-1,signals.shape[1]-1)
                signal = torch.Tensor(signal.float()).to(device)

                optimizer.zero_grad()
                prediction, mus = generator_model(signal)
                reconstruction_loss = loss_criterion(prediction, original.to(device))
                epoch_loss = + reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        test_loss = test_feature_generator(generator_model, valid_loader, feature_to_predict, historical)

        train_loss_trend.append(epoch_loss)
        test_loss_trend.append(test_loss)
        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', epoch_loss)
            print('Test ===>loss: ', test_loss)
    print('***** Training feature %d *****'%(feature_to_predict))
    print('Test loss: ', test_loss)
    # Save model and results
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    if historical:
        torch.save(generator_model.state_dict(), './ckpt/feature_%d_generator.pt'%(feature_to_predict))
    else:
        torch.save(generator_model.state_dict(), './ckpt/feature_%d_generator_nohist.pt'%(feature_to_predict))
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig('generator_train_loss.png')


def test_feature_generator(model, test_loader, feature_to_predict, historical=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    for i, (signals, labels) in enumerate(test_loader):
        if historical:
            for t in [24]:# range(10, signals.shape[2]):
                label = signals[:, feature_to_predict, t].contiguous().view(-1, 1)
                signal = torch.cat((signals[:, :feature_to_predict, t], signals[:, feature_to_predict + 1:, t]), 1)
                signal = signal.contiguous().view(-1, signals.shape[1] - 1)
                signal = torch.Tensor(signal.float()).to(device)
                past = signals[:, :, :t]
                prediction, mus = model(signal, past)
                loss = torch.nn.MSELoss()(prediction, label.to(device))
                test_loss = + loss.item()
        else:
            original = signals[:, feature_to_predict, :].contiguous().view(-1, 1)
            signal = torch.cat((signals[:, :feature_to_predict, :], signals[:, feature_to_predict + 1:, :]), 1).permute(0, 2,1)
            signal = signal.contiguous().view(-1, signals.shape[1] - 1)
            signal = torch.Tensor(signal.float()).to(device)
            prediction, mus = model(signal)
            loss = torch.nn.MSELoss()(prediction, original.to(device))
            test_loss = + loss.item()
    return test_loss




#data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, path='./data_generator/data/simulated_data')
#p_data, train_loader, valid_loader, test_loader = load_data(batch_size, './data_generator/data')
#generator = FeatureGenerator(3)
#train_feature_generator(generator, train_loader, valid_loader)
