import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

#from pydlm import dlm, autoReg

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
                                                 torch.nn.Linear(200, self.prediction_size*2))
            else:
                self.predictor = torch.nn.Sequential(torch.nn.Linear(f_size, 50),
                                                 torch.nn.Tanh(),
                                                 torch.nn.BatchNorm1d(num_features=50),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(50, self.prediction_size*2))

        else:
            if self.data=='mimic' or self.data=='ghg':
                self.predictor = torch.nn.Sequential(torch.nn.Linear(self.feature_size-1, 200),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=200),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(200, self.prediction_size*2))
            else:
                self.predictor = torch.nn.Sequential(torch.nn.Linear(self.feature_size-1, 50),
                                                 torch.nn.Tanh(),
                                                 torch.nn.BatchNorm1d(num_features=50),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(50, self.prediction_size*2))

    def forward(self, x, past, sig_ind=0, *args):
        """
        Sample full observation at t, given past information
        :param x: observation at time t
        :param past: All observations upto time t
        :param sig_ind: Index of the feature to investigate
        :return: full sample at time t
        """
        if len(x.shape) is 1:
            x = x.unsqueeze(0)
        known_signal = torch.cat((x[:, :sig_ind], x[:, sig_ind + 1:]), 1).to(self.device)
        x = torch.cat((x[:, :sig_ind], x[:, sig_ind + 1:]), 1).to(self.device)
        if self.hist:
            past = past.permute(2, 0, 1)
            prev_state = torch.zeros([1, past.shape[1], self.hidden_size]).to(self.device)
            all_encoding, encoding = self.rnn(past.to(self.device), prev_state)
            if self.conditional:
                x = torch.cat((encoding.view(encoding.size(1),-1), x), 1)
            else:
                x = encoding.view(encoding.size(1),-1)
        mu_std = self.predictor(x)
        mu = mu_std[:,0:mu_std.shape[1]//2]
        std = mu_std[:, mu_std.shape[1]//2:]
        reparam_samples = mu + std*torch.randn_like(mu).to(self.device)
        full_sample = torch.cat([known_signal[:, 0:sig_ind], reparam_samples, known_signal[:, sig_ind:]], 1)
        return full_sample, mu


class JointFeatureGenerator(torch.nn.Module):
    def __init__(self, feature_size, latent_size=100, prediction_size=1, seed=random.seed('2019'), **kwargs):
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
        super(JointFeatureGenerator, self).__init__()
        self.seed = seed
        self.feature_size = feature_size
        self.hidden_size = feature_size*2
        self.latent_size = latent_size
        self.prediction_size = prediction_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        non_lin = kwargs["non_linearity"] if "non_linearity" in kwargs.keys() else torch.nn.Tanh()
        self.data=kwargs['data'] if 'data' in kwargs.keys() else 'mimic'

        # Generates the parameters of the distribution
        self.rnn = torch.nn.GRU(self.feature_size, self.hidden_size)
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.normal(self.rnn.__getattr__(p), 0.0, 0.02)

        if self.data=='mimic' or self.data=='ghg':
            self.dist_predictor = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 100),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=100),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(100, self.latent_size*2))
        else:
            self.dist_predictor = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 10),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=10),
                                                 #torch.nn.Dropout(0.5),
                                                 torch.nn.Linear(10, self.latent_size*2), non_lin)

        self.cov_generator = torch.nn.Sequential(torch.nn.Linear(self.latent_size, 10),#+self.hidden_size, 100),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=10),
                                                 torch.nn.Linear(10, self.feature_size*self.feature_size),torch.nn.ReLU())
        self.mean_generator = torch.nn.Sequential(torch.nn.Linear(self.latent_size, 10),#+self.hidden_size, 100),
                                                 non_lin,
                                                 torch.nn.BatchNorm1d(num_features=10),
                                                 torch.nn.Linear(10, self.feature_size))

    def forward(self, x, past, sig_ind, method):
        """
        Sample full observation at t, given past information
        :param x: observation at time t
        :param past: All observations upto time t
        :param sig_ind: Index of the feature to investigate
        :param cond_one: Determines the conditioning method. If True, the joint distribution will be conditioned on only
                        a single feature, otherwise it will be conditioned on all variables except one
        :return: full sample at time t
        """
        mean, covariance = self.likelihood_distribution(past)  # P(X_t|X_0:t-1)
        if len(x.shape) is 1:
            x = x.unsqueeze(0)
        if method=='c1':  # c1 method
            x_ind = x[:, sig_ind].to(self.device).unsqueeze(-1)
            mean_1 = torch.cat((mean[:, :sig_ind], mean[:, sig_ind + 1:]), 1).unsqueeze(-1)
            cov_1_2 = torch.cat(([covariance[:, 0:sig_ind, sig_ind], covariance[:, sig_ind + 1:, sig_ind]]),
                                1).unsqueeze(-1)
            cov_2_2 = covariance[:, sig_ind, sig_ind]
            cov_1_1 = torch.cat(([covariance[:, 0:sig_ind, :], covariance[:, sig_ind + 1:, :]]), 1)
            cov_1_1 = torch.cat(([cov_1_1[:, :, 0:sig_ind], cov_1_1[:, :, sig_ind + 1:]]), 2)
            mean_cond = mean_1 + torch.bmm(cov_1_2, (x_ind - mean[:, sig_ind]).unsqueeze(-1)) / cov_2_2
            covariance_cond = cov_1_1 - torch.bmm(cov_1_2, torch.transpose(cov_1_2, 2, 1)) / cov_2_2
            likelihood = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean_cond.squeeze(-1),
                                                                               covariance_matrix=covariance_cond)
            sample = likelihood.rsample()
            full_sample = torch.cat([sample[:,0:sig_ind], x_ind, sample[:,sig_ind:]], 1)
            return full_sample, mean[:,sig_ind]
        elif method=='m1':  # m1 method
            known_signal = torch.cat((x[:, :sig_ind], x[:, sig_ind + 1:]), 1).to(self.device)
            return torch.cat((known_signal[:, 0:sig_ind], mean[:,sig_ind].unsqueeze(-1), known_signal[:, sig_ind:]), 1), mean[:,sig_ind]
        elif method=='inform':
            return torch.cat((mean[:,:sig_ind], x[:, sig_ind].unsqueeze(-1), mean[:,sig_ind+1:]), 1), mean[:,sig_ind]
        elif method=='old':
            x = torch.cat((x[:, :sig_ind], x[:, sig_ind + 1:]), 1).to(self.device)
            margianl_cov = torch.cat(([covariance[:, :, 0:sig_ind], covariance[:, :, sig_ind + 1:]]), 2)
            margianl_cov = torch.cat(([margianl_cov[:, 0:sig_ind, :], margianl_cov[:, sig_ind + 1:, :]]), 1)
            cov_i_i = torch.cat((covariance[:, sig_ind, :sig_ind], covariance[:, sig_ind, sig_ind + 1:]), 1).view(len(covariance), 1, -1)
            mean_i = torch.cat((mean[:, :sig_ind], mean[:, sig_ind + 1:]), 1)
            mean_cond = mean[:, sig_ind] + torch.bmm(torch.bmm(cov_i_i, torch.inverse(margianl_cov)),
                                                     (x - mean_i).unsqueeze(-1))
            covariance_cond = covariance[:, sig_ind, sig_ind] - torch.bmm(torch.bmm(cov_i_i, torch.inverse(margianl_cov)),
                                                                          torch.transpose(cov_i_i, 1, 2))
            likelihood = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean_cond,
                                                                                    covariance_matrix=covariance_cond)
            sample = likelihood.rsample()
            full_sample = torch.cat([x[:, 0:sig_ind], sample[0], x[:, sig_ind:]], 1)
            return full_sample, mean[:,sig_ind]

    def likelihood_distribution(self, past):
        past = past.permute(2, 0, 1)
        all_encoding, encoding = self.rnn(past.to(self.device))
        H = encoding.view(encoding.size(1),-1)
        # Find the distribution of the latent variable Z
        mu_std = self.dist_predictor(H)
        mu = mu_std[:,:mu_std.shape[1]//2]
        std = mu_std[:, mu_std.shape[1]//2:]
        # sample Z from the distribution
        Z = mu + std*torch.randn_like(mu).to(self.device)
        # Z_H = torch.cat((Z, H), 1)
        # Generate the distribution P(X|H,Z)
        mean = self.mean_generator(Z)
        cov_noise = (torch.eye(self.feature_size).unsqueeze(0).repeat(len(Z), 1, 1) * 1e-5).to(self.device)
        A = self.cov_generator(Z).view(-1, self.feature_size, self.feature_size)
        covariance = torch.bmm(A, torch.transpose(A, 1, 2)) + cov_noise
        return mean, covariance

    def forward_joint(self, past):
        mean, covariance = self.likelihood_distribution(past)
        likelihood = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)
        return likelihood.rsample()


class DLMGenerator(torch.nn.Module):
    def __init__(self, feature_size, hidden_size=800, prediction_size=1, seed=random.seed('2019'), **kwargs):
        """ Dynamic Linear Model for generating future observations in patient records
        """
        super(DLMGenerator, self).__init__()
        self.seed = seed
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.prediction_size = prediction_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.theta = torch.randn(self.feature_size*self.feature_size).to(self.device)
        # non_lin = kwargs["non_linearity"] if "non_linearity" in kwargs.keys() else torch.nn.ReLU()
        self.data = kwargs['data'] if 'data' in kwargs.keys() else 'mimic'

        self.transition = torch.nn.Sequential(torch.nn.Linear(self.feature_size*self.feature_size, self.feature_size*self.feature_size))
                                             # torch.nn.BatchNorm1d(num_features=hidden_size),
                                             # non_lin,
                                             # torch.nn.Dropout(0.5),
                                             # torch.nn.Linear(hidden_size, self.feature_size*self.feature_size))

    def forward(self, x, past, sig_ind=0):
        self.theta = self.transition(self.theta.view(1,-1)).view(self.feature_size, self.feature_size) + torch.randn_like(self.theta).to(self.device)*0.01
        prev_state = past[:,:,-1].to(self.device)
        next_state = torch.matmul(prev_state, self.theta)
        mu = next_state[:,sig_ind]
        reparam_samples = mu + torch.randn_like(mu).to(self.device)*0.1
        return reparam_samples, mu

    def forward_joint(self,past):
        self.theta = self.transition(self.theta.view(1,-1)) + torch.randn_like(self.theta).to(self.device)*0.01
        prev_state = past[:,:,-1].to(self.device)
        next_state = torch.matmul(prev_state, self.theta.view(self.feature_size, self.feature_size)) + torch.randn_like(prev_state).to(self.device)*0.01
        return next_state


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

    def forward(self, x_T, x_past, sig_ind, method):
        mu = x_past[:,sig_ind,-1]
        next_obs = mu + torch.randn_like(mu).to(self.device)*0.1
        return next_obs, mu


def save_ckpt(generator_model, fname, data):
    # Save model and results
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists(os.path.join('./ckpt',data)):
        os.mkdir(os.path.join('./ckpt',data))
    torch.save(generator_model.state_dict(), fname)


def train_feature_generator(generator_model, train_loader, valid_loader, generator_type, feature_to_predict=1, n_epochs=30, historical=False, **kwargs):
    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator_model.to(device)
    data=generator_model.data
    if data=='mimic':
        feature_map = feature_map_mimic
    elif 'simulation' in data:
        feature_map = ['0','1','2']

    # Overwrite default learning parameters if values are passed
    default_params = {'lr':0.0001, 'weight_decay':1e-3, 'generator_type':'RNN_generator'}
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
    best_loss=1000000
    for epoch in range(n_epochs + 1):
        generator_model.train()
        epoch_loss = 0
        for i, (signals, _) in enumerate(train_loader):
            if historical:
                #for t in [np.random.randint(low=24, high=45)]:
                for t in [int(tt) for tt in np.logspace(1.2,np.log10(signals.shape[2]),num=num)]:
                    label = signals[:, feature_to_predict, t:t+generator_model.prediction_size].contiguous().view(-1, generator_model.prediction_size)
                    # signal = torch.cat((signals[:, :feature_to_predict, t], signals[:, feature_to_predict + 1:, t]), 1)
                    # signal = signal.contiguous().view(-1, signals.shape[1] - 1)
                    signal = torch.Tensor(signals[:, :, t].float()).to(device)
                    past = signals[:,:,:t]
                    optimizer.zero_grad()
                    prediction, mus = generator_model(signal, past)
                    reconstruction_loss = loss_criterion(prediction, label.to(device))
                    epoch_loss = epoch_loss + reconstruction_loss.item()
                    reconstruction_loss.backward()
                    optimizer.step()

            else:
                original = signals[:,feature_to_predict,:].contiguous().view(-1,1)
                # signal = torch.cat((signals[:,:feature_to_predict,:], signals[:,feature_to_predict+1:,:]), 1).permute(0,2,1)
                # signal = signal.contiguous().view(-1,signals.shape[1]-1)
                signal = torch.Tensor(signals[:, :, t].float()).to(device)

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
        if historical:
            fname=os.path.join('./ckpt', data, '%s_%s.pt'%(feature_map[feature_to_predict], generator_type))
        else:
            fname=os.path.join('./ckpt', data, '%s_%s_nohist.pt'%(feature_map[feature_to_predict], generator_type))

        if test_loss<best_loss:
            best_loss = test_loss
            best_epoch = epoch
            save_ckpt(generator_model, fname,data)
            print('saved ckpt:in epoch', epoch)

        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            if historical:
                print('Training ===>loss: ', epoch_loss/((i+1)*num))
            else:
                print('Training ===>loss: ', epoch_loss/(i+1))
            print('Test ===>loss: ', test_loss)
    print('***** Training feature %d *****'%(feature_to_predict))
    print('Test loss: ', test_loss)

    '''
    # Save model and results
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists(os.path.join('./ckpt',data)):
        os.mkdir(os.path.join('./ckpt',data))
    if historical:
        torch.save(generator_model.state_dict(), os.path.join('./ckpt', data, '%s_%s.pt'%(feature_map[feature_to_predict], generator_type)))
    else:
        torch.save(generator_model.state_dict(),  os.path.join('./ckpt', data, '%s_%s_nohist.pt'%(feature_map[feature_to_predict], generator_type)))
    '''

    plt.figure(feature_to_predict)
    ax = plt.gca()
    plt.plot(train_loss_trend[:best_epoch], label='Train loss:Feature %d'%(feature_to_predict+1))
    plt.plot(test_loss_trend[:best_epoch], label='Validation loss: Feature %d'%(feature_to_predict+1))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./plots/'+ data):
        os.mkdir('./plots/' + data)
    if data == 'mimic':
        plt.title('%s Generator Loss'%(feature_map_mimic[feature_to_predict]),fontweight='bold',fontsize=12)
        plt.legend()
        plt.savefig('./plots/%s/%s_%s_loss.pdf'%(data, feature_map_mimic[feature_to_predict], generator_type))
    else:
        plt.title('feature %d Generator Loss'%(feature_to_predict),fontweight='bold',fontsize=12)
        plt.legend()
        plt.savefig('./plots/%s/feature_%d_%s_loss.pdf'%(data, feature_to_predict, generator_type))


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
                # signal = torch.cat((signals[:, :feature_to_predict, t], signals[:, feature_to_predict + 1:, t]), 1)
                # signal = signal.contiguous().view(-1, n_features - 1)
                signal = torch.Tensor(signals[:, :, t].float()).to(device)
                past = signals[:, :, :t]
                prediction, mus = model(signal, past)
                loss = torch.nn.MSELoss()(prediction, label.to(device))
                test_loss = + loss.item()
        else:
            original = signals[:, feature_to_predict, :].contiguous().view(-1, 1)
            # signal = torch.cat((signals[:, :feature_to_predict, :], signals[:, feature_to_predict + 1:, :]), 1).permute(0, 2,1)
            # signal = signal.contiguous().view(-1, n_features - 1)
            signal = torch.Tensor(signals[:, :, t].float()).to(device)
            prediction, mus = model(signal)
            loss = torch.nn.MSELoss()(prediction, original.to(device))
            test_loss += loss.item()
    if historical:
        test_loss = test_loss/((i+1)*len(tvec))
    return test_loss


def train_joint_feature_generator(generator_model, train_loader, valid_loader, generator_type, feature_to_predict=1, n_epochs=30, **kwargs):
    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator_model.to(device)
    data = generator_model.data
    generator_model.train()
    if data=='mimic':
        feature_map = feature_map_mimic
    elif 'simulation' in data:
        feature_map = ['0','1','2']

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
        num = 3

    best_loss = 1000000
    
    fname=os.path.join('./ckpt', data, '%s.pt'%(generator_type))

    for epoch in range(n_epochs + 1):
        generator_model.train()
        epoch_loss = 0
        for i, (signals, _) in enumerate(train_loader):
            # for t in [np.random.randint(low=24, high=45)]:
            for t in [int(tt) for tt in np.logspace(1.2, np.log10(signals.shape[2]-1), num=num)]:
                label = signals[:, :, t:t + generator_model.prediction_size].contiguous().view(signals.shape[0], signals.shape[1])
                optimizer.zero_grad()
                prediction = generator_model.forward_joint(signals[:, :, :t])
                reconstruction_loss = loss_criterion(prediction, label.to(device))
                epoch_loss = epoch_loss + reconstruction_loss.item()
                reconstruction_loss.backward(retain_graph=True)
                optimizer.step()

        test_loss = test_joint_feature_generator(generator_model, valid_loader)
        # train_loss_trend.append(epoch_loss / ((i + 1) * num))
        train_loss = test_joint_feature_generator(generator_model, train_loader)
        train_loss_trend.append(train_loss)

        test_loss_trend.append(test_loss)
        if test_loss<best_loss:
            best_loss = test_loss
            best_epoch = epoch
            print('saving ckpt')
            save_ckpt(generator_model, fname, data)

        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', train_loss)
            print('Test ===>loss: ', test_loss)
    print('***** Joint generator test loss *****', test_loss)
    #save_ckpt(generator_model,fname,data)

    '''
    # Save model and results
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists('./ckpt'+ data):
        os.mkdir('./ckpt' + data)
    torch.save(generator_model.state_dict(), './ckpt/%s/%s.pt'%(data, generator_type))
    '''

    plt.figure(feature_to_predict)
    ax = plt.gca()
    plt.plot(train_loss_trend[:best_epoch], label='Train loss:Feature %d'%(feature_to_predict+1))
    plt.plot(test_loss_trend[:best_epoch], label='Validation loss: Feature %d'%(feature_to_predict+1))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.title('Generator Loss', fontweight='bold', fontsize=12)
    plt.legend()
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./plots/'+ data):
        os.mkdir('./plots/' + data)
    plt.savefig('./plots/%s/generator_loss_%s.pdf'%(data, generator_type))


def test_joint_feature_generator(model, test_loader):
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
        for t in tvec:
            label = signals[:, :, t:t+model.prediction_size].contiguous().view(signals.shape[0], signals.shape[1])
            prediction = model.forward_joint(signals[:, :, :t])
            loss = torch.nn.MSELoss()(prediction, label.to(device))
            test_loss += loss.item()

    return test_loss/(i+1)
