import numpy as np
import pickle
import random
import torch
import os
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree

len_of_stay = 48


class PatientData():
    """Dataset of patient vitals, demographics and lab results
    Args:
        root: Root directory of dataset the pickled dataset
        train_ratio: train/test ratio
        shuffle: Shuffle dataset before separating train/test
    """

    def __init__(self, root, train_ratio=0.8, shuffle=False, random_seed='1234', transform="normalize"):
        self.data_dir = os.path.join(root, 'patient_vital_preprocessed.pkl')
        self.train_ratio = train_ratio
        self.random_seed = random.seed(random_seed)

        if not os.path.exists(self.data_dir):
            raise RuntimeError('Dataset not found')
        with open(self.data_dir, 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(root,'patient_interventions.pkl'), 'rb') as f:
            self.intervention = pickle.load(f)
        if shuffle:
            inds = np.arange(len(self.data))
            random.shuffle(inds)
            self.data = [self.data[i] for i in inds]
            self.intervention = np.array([self.intervention[i,:,:] for i in inds])
            # random.shuffle(self.data)
        self.feature_size = len(self.data[0][0])
        self.n_train = int(len(self.data) * self.train_ratio)
        self.n_test = len(self.data) - self.n_train
        self.train_data = np.array([x for (x, y, z) in self.data[0:self.n_train]])
        self.test_data = np.array([x for (x, y, z) in self.data[self.n_train:]])
        self.train_label = np.array([y for (x, y, z) in self.data[0:self.n_train]])
        self.test_label = np.array([y for (x, y, z) in self.data[self.n_train:]])
        self.train_missing = np.array([np.mean(z) for (x, y, z) in self.data[0:self.n_train]])
        self.test_missing = np.array([np.mean(z) for (x, y, z) in self.data[self.n_train:]])
        self.train_intervention = self.intervention[0:self.n_train,:,:]
        self.test_intervention = self.intervention[self.n_train:,:,:]
        if transform == "normalize":
            self.normalize()

    def __getitem__(self, index):
        signals, target = self.data[index]
        return signals, target

    def __len__(self):
        return len(self.data)

    def normalize(self):
        """ Calculate the mean and std of each feature from the training set
        """
        d = [x.T for x in self.train_data]
        d = np.stack(d, axis=0)
        self.feature_max = np.tile(np.max(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_min = np.tile(np.min(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_means = np.tile(np.mean(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_std = np.tile(np.std(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide='ignore', invalid='ignore')
        self.train_data = np.array(
           [np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std) for
            x in self.train_data])
        self.test_data = np.array(
           [np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std) for
            x in self.test_data])
        # self.train_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.train_data])
        # self.test_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.test_data])


class NormalPatientData(PatientData):
    """ Data class for the generator model that only includes patients who survived in the ICU
    """
    def __init__(self, root, train_ratio=0.8, shuffle=True, random_seed='1234', transform="normalize"):
        self.data_dir = os.path.join(root, 'patient_vital_preprocessed.pkl')
        self.train_ratio = train_ratio
        self.random_seed = random.seed(random_seed)

        if not os.path.exists(self.data_dir):
            raise RuntimeError('Dataset not found')
        with open(self.data_dir, 'rb') as f:
            self.data = pickle.load(f)
        if shuffle:
            random.shuffle(self.data)
        self.feature_size = len(self.data[0][0])
        self.n_train = int(len(self.data) * self.train_ratio)
        self.n_test = len(self.data) - self.n_train

        self.train_data = np.array([x for (x, y, z) in self.data[0:self.n_train] if y == 1])
        self.test_data = np.array([x for (x, y, z) in self.data[self.n_train:]])
        self.train_missing_samples = np.array([z for (x, y, z) in self.data[0:self.n_train] if y == 1])
        self.test_missing_samples = np.array([z for (x, y, z) in self.data[self.n_train:]])
        self.test_label = np.array([y for (x, y, z) in self.data[self.n_train:]])
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)
        self.train_label = np.zeros((len(self.train_data),1))
        self.train_missing = np.array([np.mean(z) for (x, y, z) in self.data[0:self.n_train] if y == 1])
        self.test_missing = np.array([np.mean(z) for (x, y, z) in self.data[self.n_train:]])
        if transform == "normalize":
            self.normalize()


class DeepKnn:
    """
    Look for the nearest neighbors (from training samples) of a test instance in the encoding space and
    determine prediction certainty based on the neighbors' labels
    """
    def __init__(self, model, training_samples, training_labels, device):
        self.model = model
        self.encoder_model = list(self.model.children())[0]
        self.device = device
        self.training_samples = self.encoder_model(torch.Tensor(training_samples).permute(2, 0, 1).to(self.device))
        self.training_samples = self.training_samples.detach().cpu().numpy()
        self.training_labels = training_labels
        self.encoding_tree = KDTree(self.training_samples, leaf_size=2)

    def find_neighbors(self, sample, n_nearest_neighbors=10):
        sample = self.encoder_model(torch.Tensor(sample).permute(2, 0, 1).to(self.device))
        sample = sample.detach().cpu().numpy()
        dists, indxs = self.encoding_tree.query(sample, k=n_nearest_neighbors)
        neighbors_labels = self.training_labels[indxs]
        return neighbors_labels

    def evaluate_confidence(self, sample, sample_label, n_nearest_neighbors=10, verbose=True):
        knn_labels = self.find_neighbors(sample, n_nearest_neighbors)
        matches = np.where(knn_labels==sample_label, 1, 0)
        prediction = self.model(torch.Tensor(sample).permute(2, 0, 1).to(self.device))
        if verbose:
            print('Sample true label: ', sample_label)
            print('Sample predicted risk: ', prediction.item())
            print('Closest neighbors labels: ', knn_labels.reshape(-1,))
            print('Confidence: ', np.count_nonzero(matches)/float(n_nearest_neighbors))
        return knn_labels


class EncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, rnn="GRU", regres=True, bidirectional=False, return_all=False,
                 seed=random.seed('2019')):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
        self.regressor = nn.Sequential(nn.BatchNorm1d(num_features=self.hidden_size),
                                       nn.Dropout(0.5),
                                       nn.Linear(self.hidden_size, 1),
                                       nn.Sigmoid())

    def forward(self, input, past_state=None):
        input = input.permute(2, 0, 1).to(self.device)
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, input.shape[1], self.hidden_size]).to(self.device)
        if self.rnn_type == 'GRU':
            all_encodings, encoding = self.rnn(input, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))
        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                #print('before: ', all_encodings[-1,-1,:])
                reshaped_encodings = all_encodings.view(all_encodings.shape[1]*all_encodings.shape[0],-1)
                #print('after: ', reshaped_encodings[-1:,:].data.cpu().numpy())
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0],-1))
        else:
            return encoding.view(encoding.shape[1], -1)


class RnnVAE(nn.Module):
    def __init__(self, feature_size, hidden_size, bidirectional=False,
                 seed=random.seed('2019')):
        super(RnnVAE, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # q(Zt|X0:t)
        self.encoder = nn.GRU(self.feature_size, 2*self.hidden_size, bidirectional=bidirectional)
        # P(Xt|Zt)
        self.decoder = nn.Sequential(nn.Linear(self.hidden_size,400),
                                     nn.ReLU(),
                                     nn.Linear(400,self.feature_size),
                                     nn.Sigmoid())

    def encode(self, input):
        input = input.permute(2, 0, 1)
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
        past_state = torch.zeros([1, input.shape[1], self.hidden_size*2]).to(self.device)
        _, encoding= self.encoder(input, past_state)
        mu = nn.ReLU()(encoding[:,:,:self.hidden_size]).view(-1,self.hidden_size)
        logvar = nn.ReLU()(encoding[:,:,self.hidden_size:]).view(-1,self.hidden_size)
        z = self.reparameterize(mu,logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return z


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.encoding_size = hidden_size
        self.rnn = nn.GRUCell(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, encoding, out_len, past_state=None):
        output = torch.zeros([out_len, encoding.shape[0], self.output_size])
        if not past_state:
            past_state = torch.zeros(encoding.shape).to(self.device)
        for i in range(out_len, 0, -1):
            print(encoding.shape, past_state.shape)
            encoding = self.rnn(encoding, past_state)
            past_state = nn.Softmax()(self.out(past_state))
            output[i - 1, :, :] = past_state
        return output


class LR(nn.Module):
    def __init__(self, feature_size):
        super(LR, self).__init__()
        self.feature_size = feature_size
        self.net = nn.Sequential(nn.Linear(self.feature_size, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = x.mean(dim=2).reshape((x.shape[0], -1))
        if len(x.shape) == 3:
            x = x.view(-1, self.feature_size)
        risk = (self.net(x))
        return risk


class RiskPredictor(nn.Module):
    def __init__(self, encoding_size):
        super(RiskPredictor, self).__init__()
        self.encoding_size = encoding_size
        self.net = nn.Sequential(nn.Linear(self.encoding_size, 500),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),
                                 nn.Linear(500, 1))

    def forward(self, x):
        risk = nn.Sigmoid()(self.net(x))
        return risk


class Encoder(nn.Module):
    """ This encoder consists of a CNN and a RNN layer on top of it,
        in order to be able to handle variable length inputs
    """

    def __init__(self, in_channels, encoding_size=64):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.enc_channels = 32
        self.encoding_size = encoding_size
        self.enc_cnn = nn.Sequential(nn.Conv1d(self.in_channels, self.enc_channels, kernel_size=4),
                                     nn.MaxPool1d(kernel_size=2, stride=2),
                                     nn.ReLU())
        self.enc_lstm = nn.LSTMCell(input_size=self.enc_channels, hidden_size=self.encoding_size)

    def forward(self, signals):
        code = self.enc_cnn(signals)
        h = torch.zeros(code.shape[0], self.encoding_size)
        c = torch.zeros(code.shape[0], self.encoding_size)
        for i in range(code.shape[2]):
            h, c = self.enc_lstm(code[:, :, i].view(code.shape[0], code.shape[1]), (h, c))
        return h


class Decoder(nn.Module):
    def __init__(self, output_len, encoding_size):
        super(Decoder, self).__init__()
        self.encoding_size = encoding_size
        self.output_len = output_len
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.encoding_size, 160)
        self.dec_linear_2 = nn.Linear(160, self.output_len)

    def forward(self, code):
        out = F.relu(self.dec_linear_1(code))
        out = self.dec_linear_2(out)
        out = out.view([code.size(0), 1, self.output_len])
        return out
