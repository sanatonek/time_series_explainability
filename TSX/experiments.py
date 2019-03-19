import torch
import os
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, test
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE


class Experiment(ABC):
    def __init__(self, train_loader, valid_loader, test_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    @abstractmethod
    def run(self):
        raise RuntimeError('Function not implemented')

    def train(self, n_epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-3)
        train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment)
        # Evaluate performance on held-out test set
        _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
        print('\nFinal performance on held out test set ===> AUC: ', auc_test)


class Baseline(Experiment):
    """ Baseline mortality prediction using a logistic regressions model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, experiment='baseline'):
        super(Baseline, self).__init__(train_loader, valid_loader, test_loader)
        self.model = LR(feature_size).to(self.device)
        self.experiment = experiment

    def run(self, train):
        if train:
            self.train(n_epochs=120)
        else:
            if os.path.exists('./ckpt/' + str(self.experiment) + '.pt'):
                self.model.load_state_dict(torch.load('./ckpt/' + str(self.experiment) + '.pt'))
                _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                print('Loading model with AUC: ', auc_test)
            else:
                raise RuntimeError('No saved checkpoint for this model')


class EncoderPredictor(Experiment):
    """ Baseline mortality prediction using an encoder to encode patient status, and a risk predictor to predict risk of mortality
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor'):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader)
        self.state_encoder = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=False)
        self.risk_predictor = RiskPredictor(encoding_size)
        self.model = torch.nn.Sequential(self.state_encoder, self.risk_predictor)
        self.experiment = experiment

    def run(self, train):
        if train:
            self.train(n_epochs=120)
        else:
            if os.path.exists('./ckpt/' + str(self.experiment) + '.pt'):
                self.model.load_state_dict(torch.load('./ckpt/' + str(self.experiment) + '.pt'))
                _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                print('Loading model with AUC: ', auc_test)
            else:
                raise RuntimeError('No saved checkpoint for this model')


class KalmanExperiment(Experiment):
    """ Build a Deep Kalman filter to encode patient status, and use a risk predictor to
    """
    def __init__(self, train_loader, valid_loader, test_loader, feature_size, encoding_size):
        super(KalmanExperiment, self).__init__(train_loader, valid_loader, test_loader)
        self.state_encoder = RnnVAE(feature_size=feature_size, hidden_size=encoding_size)
        self.risk_predictor = RiskPredictor(encoding_size=encoding_size)
        self.model = torch.nn.Sequential(self.state_encoder, self.risk_predictor)
        self.state_encoder, self.risk_predictor, self.model = self.state_encoder.to(self.device), self.risk_predictor.to(self.device), self.model.to(self.device)

    def run(self,train):
        self.train(n_iter=2)
        print(test_reconstruction(self.state_encoder, self.test_loader, self.device))

    def train(self, n_iter):
        for _ in range(n_iter):
            print("\n***** Train Encoder *****")
            train_reconstruction(self.state_encoder, self.train_loader, self.valid_loader, 80, self.device, "VAE")
            print("\n***** Train Predictor *****")
            optimizer = torch.optim.Adam(self.risk_predictor.parameters(), lr=0.0001, weight_decay=1e-3)
            train_model(self.model, self.train_loader, self.valid_loader, optimizer, 80, self.device, "VAE")

