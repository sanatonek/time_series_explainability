import torch
import os
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, test
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE
from TSX.generator import Generator, test_generator
import matplotlib.pyplot as plt
import numpy as np


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
        #self.state_encoder = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=False)
        #self.risk_predictor = RiskPredictor(encoding_size)
        #self.model = torch.nn.Sequential(self.state_encoder, self.risk_predictor)
        self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True)
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


class GeneratorExplainer(Experiment):
    """ Generating time step importance using a time series generator
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, experiment='generator_explainer'):
        super(GeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader)
        #self.state_encoder = EncoderRNN(feature_size, encoding_size, regres=False)
        #self.predictor = RiskPredictor(encoding_size)
        #self.risk_predictor = torch.nn.Sequential(self.state_encoder, self.predictor)
        self.risk_predictor = EncoderRNN(feature_size, encoding_size, rnn='GRU', regres=True)
        # Demographics are not fed into the generator model
        self.generator = Generator(feature_size-4).to(self.device)
        self.experiment = experiment

    def run(self, train):
        if train:
            self.train(n_epochs=120)
        else:
            if os.path.exists('./ckpt/risk_predictor.pt') and os.path.exists('./ckpt/generator.pt'):
                self.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
                self.generator.load_state_dict(torch.load('./ckpt/generator.pt'))
                _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.risk_predictor, self.device)
                print('Loading model with AUC: ', auc_test)
                gen_test_loss = test_generator(self.generator, self.test_loader)
                print('Generator test loss: ', gen_test_loss)
            else:
                raise RuntimeError('No saved checkpoint for this model')

            testset = list(self.test_loader.dataset)
            labels = [s[1] for s in testset]
            print(labels.index(1))
            for subject in [20]:#range(30):
                signals, label = testset[subject]
                print('Patient dead?: ', label.item())
                risk, importance, mean_predicted_risk, std_predicted_risk = self._get_importance(signals)
                t = range(12,47)
                plt.plot(t,risk, label='Risk score')
                #plt.plot(t,importance, label='Importance')
                plt.plot(t, mean_predicted_risk, label='Estimated score')
                plt.errorbar(t, importance, yerr=std_predicted_risk, marker='^', label='Time step importance')
                plt.legend()
                plt.show()

    def _get_importance(self, signal):
        risk = []
        importance = []
        mean_predicted_risk = []
        std_predicted_risk = []
        for t in range(12, 47):
            # print('\nPredicted risk score at %dth hour: '%(t),  self.risk_predictor(signals[:,0:t].view(1, signals.shape[0], t).to(self.device)).item())
            # print('Predicted risk score at %dth hour: '%(t+1), self.risk_predictor(signals[:,0:t+1].view(1, signals.shape[0], t+1).to(self.device)).item())
            r = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t + 1).to(self.device)).item()
            risk.append(r)
            # print('Predicted risk score at 19th hour (based on the generator): ', self.risk_predictor(generated_sig.to(self.device)).item())
            predicted_risk = []
            for _ in range(10):
                predicted_step, _ = self.generator(signal[:-4, 0:t].view(1, signal.shape[0] - 4, t).to(self.device))
                generated_sig = signal[:, 0:t + 1].view(1, signal.shape[0], t + 1).clone()
                generated_sig[0, :-4, -1] = predicted_step
                predicted_risk.append(self.risk_predictor(generated_sig.to(self.device)).item())
            predicted_risk = np.array(predicted_risk)
            mean_imp = np.mean(predicted_risk,0)
            std_imp = np.std(predicted_risk, 0)
            mean_predicted_risk.append(mean_imp)
            std_predicted_risk.append(std_imp)
            importance.append(abs(mean_imp-r))
        return risk, importance, mean_predicted_risk, std_predicted_risk


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

