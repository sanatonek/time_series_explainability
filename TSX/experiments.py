import torch
import os
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, test, logistic
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE
from TSX.generator import Generator, test_generator, FeatureGenerator, test_feature_generator, train_feature_generator
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


feature_map = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

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
    ## TODO: This function currently doesn't work for simulation data
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, simulation=False, experiment='generator_explainer'):
        super(GeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader)
        #self.state_encoder = EncoderRNN(feature_size, encoding_size, regres=False)
        #self.predictor = RiskPredictor(encoding_size)
        #self.risk_predictor = torch.nn.Sequential(self.state_encoder, self.predictor)
        self.risk_predictor = EncoderRNN(feature_size, encoding_size, rnn='GRU', regres=True)
        self.simulation = simulation
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


class FeatureGeneratorExplainer(Experiment):
    """ Generating feature importance over time using a generative model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, historical=False, simulation=False,  experiment='feature_generator_explainer'):
        super(FeatureGeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader)
        self.generator = FeatureGenerator(feature_size, historical).to(self.device)
        self.feature_size = feature_size
        self.experiment = experiment
        self.historical = historical
        self.simulation = simulation
        if simulation:
            self.risk_predictor = lambda signal,t:logistic(.5 * signal[0, t] * signal[0, t] + 0.5 * signal[1,t] * signal[1,t] + 0.5 * signal[2, t] * signal[2, t])
        else:
            self.risk_predictor = EncoderRNN(feature_size, hidden_size=100, rnn='GRU', regres=True)

    def run(self, train):
        if train:
            self.train(self.feature_size, n_epochs=30)
        else:
            if os.path.exists('./ckpt/feature_0_generator.pt'):
                if not self.simulation:
                    self.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
                    self.risk_predictor = self.risk_predictor.to(self.device)
                    self.risk_predictor.eval()
                #gen_test_loss = test_feature_generator(self.generator, self.test_loader, 1)
                #print('Generator test loss: ', gen_test_loss)
            else:
                raise RuntimeError('No saved checkpoint for this model')

            with open(os.path.join('./data_generator/data/simulated_data/thresholds_test.pkl'), 'rb') as f:
                th = pkl.load(f)

            testset = list(self.test_loader.dataset)
            label = np.array([x[1] for x in testset])
            # dead = np.where(label==1)[0]
            # sub = np.random.choice(dead, 10, replace=False)
            samples_to_analyse = [2313, 4258, 3048,3460,881,188,3845,454]#,58,218,86,45]

            ## Sensitivity analysis as a baseline
            signal = torch.stack([sample[0] for i,sample in enumerate(testset) if i in samples_to_analyse])
            if not self.simulation:
                self.risk_predictor.train()
                signal = torch.Tensor(signal).to(self.device).requires_grad_()
                out = self.risk_predictor(signal)
                out[0].backward(retain_graph=True)
                #print('Feature importance using sensitivity analysis:\n', torch.mean(signal.grad.data,2)[:, 0:3])
                self.risk_predictor.eval()

            print('\n********** Visualizing a few samples **********')
            for sub_ind, subject in enumerate(samples_to_analyse):#range(30):
                f, (ax1, ax2) = plt.subplots(2, sharex=True)
                signals, label_o = testset[subject]
                # print('Change thresholds: ', th[subject])
                print('Subject ID: ', subject)
                print('Did this patient die? ', {1:'yes',0:'no'}[label_o.item()])
                t = np.arange(47)
                importance = np.zeros((28,47))
                mean_predicted_risk = np.zeros((28,47))
                std_predicted_risk = np.zeros((28,47))
                max_imp = []
                for i, sig_ind in enumerate(range(0,28)):#[5,6,15,26,25]):#range(self.feature_size):
                    self.generator.load_state_dict(torch.load('./ckpt/feature_%d_generator.pt'%(sig_ind)))
                    label, importance[i,:], mean_predicted_risk[i,:], std_predicted_risk[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind)
                    max_imp.append((i,max(mean_predicted_risk[i,:])))

                ## Pick the most influential signals and plot thir importance over time
                max_imp.sort(key=lambda pair: pair[1], reverse=True)
                for sig in range(10):
                    ind = max_imp[sig][0]
                    # plt.plot(t, mean_predicted_risk, label='Estimated score imputing %d'%(sig_ind))
                    # ax2.errorbar(t, importance[ind,:], yerr=std_predicted_risk[ind,:], marker='^', label='%s importance'%(feature_map[ind]))
                    ax2.errorbar(t, importance[ind,:], label='%s'%(feature_map[ind]))
                    ax1.plot(np.array(signals[ind,:]), label='%s'%(feature_map[ind]))
                    # plt.plot(abs(signal.grad.data[sub_ind,i,:].cpu().detach().numpy()*1e+2), label='Feature %s Sensitivity analysis'%(feature_map[sig_ind]))
                ax1.plot(t, np.array(label), '--', label='Risk score')
                ax1.legend()
                ax2.legend()
                ax1.grid()
                ax2.grid()
                ax1.set_title('Time series signals')
                ax2.set_title('Signal importance')
                plt.show()


            # tp = [[]]*8
            # fn = [[]]*8
            # tn = [[]]*8
            # fp = [[]]*8
            # for i,sig in enumerate(range(20,28)):
            #     self.generator.load_state_dict(torch.load('./ckpt/feature_%d_generator.pt'%(sig)))
            #     for signals,label in testset:
            #         pred, importance, mean_predicted_risk, std_predicted_risk = self._get_feature_importance(signals,
            #                                                                                                   sig_ind=sig)
            #         imp = np.mean(mean_predicted_risk)
            #         if label==1:
            #             if pred[-1]>.5:
            #                 # True positive
            #                 tp[i].append(imp)
            #             if pred[-1] <= .5:
            #                 # False negative
            #                 fn[i].append(imp)
            #         if label==0:
            #             if pred[-1] >.5:
            #                 # False positive
            #                 fp[i].append(imp)
            #             if pred[-1] <= .5:
            #                 # True negative
            #                 tn[i].append(imp)
            # tp = [np.mean(imps) for imps in tp]
            # fn = [np.mean(imps) for imps in fn]
            # tn = [np.mean(imps) for imps in tn]
            # fp = [np.mean(imps) for imps in fp]
            # print("Importance of TRUE POSITIVES: ", tp)
            # print("Importance of FALSE NEGATIVES: ", fn)
            # print("Importance of TRUE NEGATIVES: ", tn)
            # print("Importance of FALSE POSITIVES: ", fp)

    def train(self, n_features, n_epochs):
        for feature_to_predict in range(0,28):#range(n_features):
            train_feature_generator(self.generator, self.train_loader, self.valid_loader, feature_to_predict, 40, self.historical)

    def _get_feature_importance(self, signal, sig_ind):
        self.generator.eval()

        risks = []
        importance = []
        mean_predicted_risk = []
        std_predicted_risk = []
        for t in range(1,signal.shape[1]):
            if self.simulation:
                risk = self.risk_predictor(signal.cpu().detach().numpy(), t)
            else:
                risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t + 1)).item()
            signal_known = torch.cat((signal[:sig_ind,t], signal[sig_ind+1:,t])).to(self.device)
            signal = signal.to(self.device)
            risks.append(risk)
            # print('Predicted risk score at 19th hour (based on the generator): ', self.risk_predictor(generated_sig.to(self.device)).item())
            predicted_risks = []
            for _ in range(10):
                prediction, _ = self.generator(signal_known.view(1,-1), signal[:, 0:t].view(1,signal.size(0),t))
                #predicted_risk = logistic(.5 * signal[0, t] * signal[0, t] + 0.5 * prediction.item() * prediction.item() + 0.5 * signal[2, t] * signal[2, t])
                predicted_signal = signal[:,0:t+1].clone()
                predicted_signal[:,t] = torch.cat((signal[:sig_ind,t], prediction.view(-1), signal[sig_ind+1:,t]),0)
                if self.simulation:
                    predicted_risk = self.risk_predictor(predicted_signal.cpu().detach().numpy(), t)
                else:
                    predicted_risk = self.risk_predictor(predicted_signal[:, 0:t + 1].view(1, predicted_signal.shape[0], t + 1).to(self.device)).item()
                predicted_risks.append(predicted_risk)
            predicted_risks = np.array(predicted_risks)
            mean_imp = np.mean(predicted_risks,0)
            std_imp = np.std(predicted_risks, 0)
            mean_predicted_risk.append(mean_imp)
            std_predicted_risk.append(std_imp)
            importance.append(abs(mean_imp-risk))
        return risks, importance, mean_predicted_risk, std_predicted_risk


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

