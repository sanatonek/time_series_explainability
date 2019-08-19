import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']


class FeatureGenerator(torch.nn.Module):
    def __init__(self, feature_size, hist=False, hidden_size=100, prediction_size=1, conditional=True, seed=random.seed('2019'), **kwargs):
        """ Conditional generator model to predict future observations
        :param feature_size: Number of features in the input
        :param hist: (boolean) If True, use previous observations in the time series to generate next observation.
                            If False, generate the sample given other dimensions at that time point
        :param hidden_size: Size of hidden units for the recurrent structure
        :param prediction_size: Number of time steps to generate
        :param conditional: (boolean) If True, use both other observations at time t as well as the history to
                            generate future observations
        :param seed: Random seed
        """
        super(FeatureGenerator, self).__init__()
        self.seed = seed
        self.hist = hist
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.conditional = conditional
        self.prediction_size = prediction_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        non_lin = kwargs["non_linearity"] if "non_linearity" in kwargs.keys() else torch.nn.ReLU()
        self.data=kwargs['data'] if 'data' in kwargs.keys() else 'mimic'

        if self.hist:
            self.rnn = torch.nn.GRU(self.feature_size, self.hidden_size)
            # f_size is the size of the input to the regressor, it equals the hidden size of
            # the recurrent model if observation is conditioned on the pas only
            # If it is also conditioned on current observations of other dimensions
            # the size will be hidden_size+number of other dimensions
            f_size = self.hidden_size
            if conditional:
                f_size = f_size + self.feature_size-1

            if self.data=='mimic' or self.data=='ghg':
                self.predictor = torch.nn.Sequential(torch.nn.Linear(f_size, 200),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=200),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(200, self.prediction_size))
            else:
                self.predictor = torch.nn.Sequential(torch.nn.Linear(f_size, 200),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=200),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(200, self.prediction_size), non_lin)

        else:
            if self.data=='mimic' or self.data=='ghg':
                self.predictor = torch.nn.Sequential(torch.nn.Linear(self.feature_size-1, 200),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=200),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(200, self.prediction_size))
            else:
                self.predictor = torch.nn.Sequential(torch.nn.Linear(self.feature_size-1, 200),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=200),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(200, self.prediction_size), non_lin)

    def forward(self, x, past, sig_ind):
        if self.hist:
            past = past.permute(2, 0, 1)
            prev_state = torch.zeros([1, past.shape[1], self.hidden_size]).to(self.device)
            all_encoding, encoding = self.rnn(past.to(self.device), prev_state)
            if self.conditional:
                x = torch.cat((encoding.view(encoding.size(1),-1), x), 1)
            else:
                x = encoding.view(encoding.size(1),-1)
        mu = self.predictor(x)
        reparam_samples = mu + torch.randn_like(mu).to(self.device)*0.1
        return reparam_samples, mu


class CarryForwardGenerator(torch.nn.Module):
    def __init__(self, feature_size, prediction_size=1, seed=random.seed('2019')):
        """ Carries on the last observation to the nest
        :param
        """
        super(CarryForwardGenerator, self).__init__()
        self.seed = seed
        self.feature_size = feature_size
        self.prediction_size = prediction_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x_T, x_past, sig_ind):
        mu = x_past[:,sig_ind,-1]
        next_obs = mu + torch.randn_like(mu).to(self.device)*0.1
        return next_obs, mu


def train_feature_generator(generator_model, train_loader, valid_loader, feature_to_predict=1, path='./ckpt/', n_epochs=30, historical=False, **kwargs):
    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator_model.to(device)
    data=generator_model.data

    # Overwrite default learning parameters if values are passed
    default_params = {'lr':0.0001, 'weight_decay':1e-3}
    for k,v in kwargs.items():
        if k in default_params.keys():
            default_params[k] = v

    parameters = generator_model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=default_params['lr'], weight_decay=default_params['weight_decay'])
    loss_criterion = torch.nn.MSELoss()

    if data=='mimic':
        num = 1
    else:
        num = 1

    for epoch in range(n_epochs + 1):
        generator_model.train()
        epoch_loss = 0
        for i, (signals, _) in enumerate(train_loader):
            if historical:
                #for t in [np.random.randint(low=24, high=45)]:
                for t in [int(tt) for tt in np.logspace(1.2,np.log10(signals.shape[2]),num=num)]:
                    label = signals[:, feature_to_predict, t:t+generator_model.prediction_size].contiguous().view(-1, generator_model.prediction_size)
                    signal = torch.cat((signals[:, :feature_to_predict, t], signals[:, feature_to_predict + 1:, t]), 1)
                    signal = signal.contiguous().view(-1, signals.shape[1] - 1)
                    signal = torch.Tensor(signal.float()).to(device)
                    past = signals[:,:,:t]
                    optimizer.zero_grad()
                    prediction, mus = generator_model(signal, past)
                    reconstruction_loss = loss_criterion(prediction, label.to(device))
                    epoch_loss = epoch_loss + reconstruction_loss.item()
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
                epoch_loss += reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        test_loss = test_feature_generator(generator_model, valid_loader, feature_to_predict, historical)
        if historical:
            train_loss_trend.append(epoch_loss/((i+1)*num))
        else:
            train_loss_trend.append(epoch_loss/(i+1))

        test_loss_trend.append(test_loss)
        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            if historical:
                print('Training ===>loss: ', epoch_loss/((i+1)*num))
            else:
                print('Training ===>loss: ', epoch_loss/(i+1))
            print('Test ===>loss: ', test_loss)
    print('***** Training feature %d *****'%(feature_to_predict))
    print('Test loss: ', test_loss)
    # Save model and results
    if not os.path.exists(path):
        os.mkdir(path)
    if historical:
        if data=='mimic':
            torch.save(generator_model.state_dict(), os.path.join(path,'%s_generator.pt'%(feature_map_mimic[feature_to_predict])))
        else:
            torch.save(generator_model.state_dict(), os.path.join(path,'%d_generator.pt'%(feature_to_predict)))
    else:
        if data=='mimic':
            torch.save(generator_model.state_dict(),  os.path.join(path,'%s_generator_nohist.pt'%(feature_map_mimic[feature_to_predict])))
        else:
            torch.save(generator_model.state_dict(),  os.path.join(path,'%d_generator_nohist.pt'%(feature_to_predict)))

    plt.figure(feature_to_predict)
    ax = plt.gca()
    plt.plot(train_loss_trend, label='Train loss:Feature %d'%(feature_to_predict+1))
    plt.plot(test_loss_trend, label='Validation loss: Feature %d'%(feature_to_predict+1))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.title('%s Generator Loss'%(feature_map_mimic[feature_to_predict]),fontweight='bold',fontsize=12)
    plt.legend()
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if data=='mimic':
        plt.savefig('./plots/%s_generator_loss.png'%(feature_map_mimic[feature_to_predict]))
    else:
        if not os.path.exists('./plots/'+ data):
            os.mkdir('./plots/'+ data)
        plt.savefig('./plots/%s/%d_generator_loss.png'%(data,feature_to_predict))


def test_feature_generator(model, test_loader, feature_to_predict, historical=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = model.data
    model.eval()
    _, n_features, signel_len = next(iter(test_loader))[0].shape
    test_loss = 0
    if data == 'mimic':
        tvec = [24]
    else:
        num = 1
        tvec = [int(tt) for tt in np.logspace(1.0,np.log10(signel_len), num=num)]
    for i, (signals, labels) in enumerate(test_loader):
        if historical:
            for t in tvec:
                label = signals[:, feature_to_predict, t:t+model.prediction_size].contiguous().view(-1, model.prediction_size)
                signal = torch.cat((signals[:, :feature_to_predict, t], signals[:, feature_to_predict + 1:, t]), 1)
                signal = signal.contiguous().view(-1, n_features - 1)
                signal = torch.Tensor(signal.float()).to(device)
                past = signals[:, :, :t]
                prediction, mus = model(signal, past)
                loss = torch.nn.MSELoss()(prediction, label.to(device))
                test_loss = + loss.item()
        else:
            original = signals[:, feature_to_predict, :].contiguous().view(-1, 1)
            signal = torch.cat((signals[:, :feature_to_predict, :], signals[:, feature_to_predict + 1:, :]), 1).permute(0, 2,1)
            signal = signal.contiguous().view(-1, n_features - 1)
            signal = torch.Tensor(signal.float()).to(device)
            prediction, mus = model(signal)
            loss = torch.nn.MSELoss()(prediction, original.to(device))
            test_loss = + loss.item()
    if historical:
        test_loss = test_loss/((i+1)*len(tvec))
    return test_loss

