import torch
import os
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, train_model_rt, test, test_rt, logistic
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE
from TSX.generator import Generator, test_generator, FeatureGenerator, test_feature_generator, train_feature_generator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pickle as pkl
import pandas as pd

xkcd_colors = mcolors.XKCD_COLORS
color_map = [list(xkcd_colors.keys())[k] for k in
             np.random.choice(range(len(xkcd_colors)), 28, replace=False)]
color_map = ['#990000', '#C20088', '#0075DC', '#993F00', '#4C005C', '#191919', '#FF0010', '#2BCE48', '#FFCC99', '#808080',
             '#740AFF', '#8F7C00', '#9DCC00', '#F0A3FF', '#94FFB5', '#FFA405', '#FFA8BB', '#426600', '#005C31', '#5EF1F2',
             '#00998F', '#E0FF66', '#003380', '#990000', '#FFFF80', '#FF5005', '#FFFF00','#FF0010', '#FFCC99']


feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

feature_map_simulation = ['var 0', 'var 1', 'var 2']

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']

class Experiment(ABC):
    def __init__(self, train_loader, valid_loader, test_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    @abstractmethod
    def run(self):
        raise RuntimeError('Function not implemented')

    def train(self, n_epochs,learn_rt=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-3)
        
        if not learn_rt:
            train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment)
            # Evaluate performance on held-out test set
            _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            print('\nFinal performance on held out test set ===> AUC: ', auc_test)
        else:
            train_model_rt(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment)


class Baseline(Experiment):
    """ Baseline mortality prediction using a logistic regressions model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, experiment='baseline'):
        super(Baseline, self).__init__(train_loader, valid_loader, test_loader)
        self.model = LR(feature_size).to(self.device)
        self.experiment = experiment

    def run(self, train):
        if train:
            self.train(n_epochs=250)
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
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor',simulation=False):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader)
        #self.state_encoder = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=False)
        #self.risk_predictor = RiskPredictor(encoding_size)
        #self.model = torch.nn.Sequential(self.state_encoder, self.risk_predictor)
        self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=simulation)
        self.experiment = experiment
        self.simulation = simulation

    def run(self, train):
        if train:
            self.train(n_epochs=80, learn_rt=self.simulation)
        else:
            if os.path.exists('./ckpt/' + str(self.experiment) + '.pt'):
                self.model.load_state_dict(torch.load('./ckpt/' + str(self.experiment) + '.pt'))

                if not self.simulation:
                    _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                else:
                    test_loss = test_rt(self.test_loader, self.model, self.device)
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
    """ Experiment for generating feature importance over time using a generative model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, patient_data, generator_hidden_size=80, prediction_size=1, historical=False, simulation=False,  experiment='feature_generator_explainer'):
        """
        :param train_loader:
        :param valid_loader:
        :param test_loader:
        :param feature_size: Size of input features
        :param patient_data:
        :param generator_hidden_size: Generator hidden size
        :param prediction_size: Size of the prediction window. AKA number of samples the generator predicts
        :param historical: (boolean) If True, use past history for the generator
        :param simulation: (boolean) If True, run experiment on simulated data
        :param experiment: Experiment name
        """
        super(FeatureGeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader)
        self.generator = FeatureGenerator(feature_size, historical, hidden_size=generator_hidden_size, prediction_size=prediction_size).to(self.device)
        if not simulation:
            self.timeseries_feature_size = feature_size - 4
        self.feature_size = feature_size
        self.input_size = feature_size
        self.patient_data = patient_data
        self.experiment = experiment
        self.historical = historical
        self.simulation = simulation
        self.prediction_size = prediction_size
        self.generator_hidden_size = generator_hidden_size
        #this is used to see fhe difference between true risk vs learned risk for simulations
        self.learned_risk = True
        trainset = list(self.train_loader.dataset)
        self.feature_dist = torch.stack([x[0] for x in trainset])
        self.feature_dist_0 = torch.stack([x[0] for x in trainset if x[1]==0])
        self.feature_dist_1 = torch.stack([x[0] for x in trainset if x[1]==1])
        if simulation:
            if self.simulation and not self.learned_risk:
                self.risk_predictor = lambda signal,t:logistic(1.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t] - 1))
            elif self.simulation and self.learned_risk:
                self.risk_predictor = EncoderRNN(feature_size,hidden_size=20,rnn='GRU',regres=True, return_all=True)
            else:
                self.risk_predictor = EncoderRNN(feature_size,hidden_size=20,rnn='GRU',regres=True, return_all=False)
#x[0,2:50]*x[0,2:50] +  np.log(x[1,2:50]+3)*x[1,2:50] +  x[2,2:50]*x[2,2:50] -0.5
            self.feature_map = feature_map_simulation
        else:
            self.risk_predictor = EncoderRNN(self.input_size, hidden_size=150, rnn='GRU', regres=True)
            self.feature_map = feature_map_mimic
        self.risk_predictor = self.risk_predictor.to(self.device)

    def run(self, train):
        """ Run feature generator experiment
        :param train: (boolean) If True, train the generators, if False, use saved checkpoints
        """
        if train:
            self.train(n_features=self.feature_size, n_epochs=150)
        else:
            if not os.path.exists('./ckpt/feature_0_generator.pt'):
                raise RuntimeError('No saved checkpoint for this model')
            else:
                if not self.simulation:
                    self.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
                    self.risk_predictor = self.risk_predictor.to(self.device)
                    self.risk_predictor.eval()
                else: #simulated data
                    if self.learned_risk:
                        self.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
                        self.risk_predictor = self.risk_predictor.to(self.device)
                        self.risk_predictor.eval()
                #gen_test_loss = test_feature_generator(self.generator, self.test_loader, 1)
                #print('Generator test loss: ', gen_test_loss)

            testset = list(self.test_loader.dataset)
            if self.simulation:
                with open(os.path.join('./data_generator/data/simulated_data/thresholds_test.pkl'), 'rb') as f:
                    th = pkl.load(f)
                #For simulated data this is the last entry - end of 48 hours that's the actual outcome
                label = np.array([x[1][-1] for x in testset])
                #print(label)
                high_risk = np.where(label>=0.5)[0]
                samples_to_analyse = np.random.choice(high_risk, 5, replace=False)


            samples_to_analyse = [4387, 481]
            ## Sensitivity analysis as a baseline
            signal = torch.stack([testset[sample][0] for sample in samples_to_analyse])
            sensitivity_analysis = np.zeros((signal.shape))

            if not self.simulation:
                self.risk_predictor.train()
                for t in range(1,signal.size(2)):
                    signal_t = torch.Tensor(signal[:,:,:t+1]).to(self.device).requires_grad_()
                    out = self.risk_predictor(signal_t)
                    for s in range(len(samples_to_analyse)):
                        out[s, 0].backward(retain_graph=True)
                        sensitivity_analysis[s,:,t] = signal_t.grad.data[s,:,t].cpu().detach().numpy()
                        signal_t.grad.data.zero_()
                self.risk_predictor.eval()
            else:
                #print(testset[0][0].shape)
                if not self.learned_risk:
                    out = np.array([np.array([self.risk_predictor(sample[0].cpu().detach().numpy(),t) for t in range(48)]) for i,sample in enumerate(testset) if i in samples_to_analyse])
                    grad_out = []
                    for kk,i in enumerate(samples_to_analyse):
                        sample = testset[i][0].cpu().detach().numpy()
                        grad_x0 = 3*out[kk,:]*(1-out[kk,:])*sample[0,:]
                        #grad_x1 = out[kk,:]*(1-out[kk,:])*(1 + np.log(sample[1,:] + 1.))
                        grad_x1 = 3*out[kk,:]*(1-out[kk,:])*sample[1,:]
                        grad_x2 = 3*out[kk,:]*(1-out[kk,:])*sample[2,:]
                        grad_out.append(np.stack([grad_x0, grad_x1, grad_x2]))
                    grad_out = np.array(grad_out)
                    sensitivity_analysis = grad_out
                else:
                    #In simulation data also get sensitivity w.r.t. a learned predictor
                    self.risk_predictor.train()
                    signal = torch.Tensor(signal).to(self.device).requires_grad_()
                    out = self.risk_predictor(signal)
                    grad_out = []
                    for i in range(out.shape[0]):
                        grad_vec = np.zeros([self.feature_size,48])
                        for t in range(48):
                            out[i,t].backward(retain_graph=True)
                            grad_vec[:,t] = signal.grad.data[i,:,t].cpu().detach().numpy()
                        grad_out.append(grad_vec)
                    grad_out = np.array(grad_out)
                    sensitivity_analysis = grad_out
                    self.risk_predictor.eval()

            print('\n********** Visualizing a few samples **********')
            self.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
            self.risk_predictor.to(self.device)
            self.risk_predictor.eval()
            signals_to_analyze = range(0,27)
            for sub_ind, subject in enumerate(samples_to_analyse):
                if self.simulation:
                    signals, label_o = testset[subject]
                    label_o = label_o[-1]
                    print('Subject ID: ', subject)
                    print('Did this patient die? ', {1:'yes',0:'no'}[label_o.item()>0.5])
                else:
                    self.plot_baseline(subject, signals_to_analyze, sensitivity_analysis[sub_ind,:,:])

    def plot_baseline(self, subject, signals_to_analyze, sensitivity_analysis_importance, retain_style=False, n_important_features=3):
        """ Plot importance score across all baseline methods
        :param subject: ID of the subject to analyze
        :param signals_to_analyze: list of signals to include in importance analysis
        :param sensitivity_analysis_importance: Importance score over time under sensitivity analysis for the subject
        :param retain_style: Plotting mode. If true, top few important signal names will be plotted at every time point
        :param n_important_features: Number of important signals to plot
        """
        testset = list(self.test_loader.dataset)
        signals, label_o = testset[subject]
        print('Subject ID: ', subject)
        print('Did this patient die? ', {1: 'yes', 0: 'no'}[label_o.item()])

        importance = np.zeros((self.timeseries_feature_size, 47))
        mean_predicted_risk = np.zeros((self.timeseries_feature_size, 47))
        std_predicted_risk = np.zeros((self.timeseries_feature_size, 47))
        importance_occ = np.zeros((self.timeseries_feature_size, 47))
        std_predicted_risk_occ = np.zeros((self.timeseries_feature_size, 47))
        importance_occ_aug = np.zeros((self.timeseries_feature_size, 47))
        std_predicted_risk_occ_aug = np.zeros((self.timeseries_feature_size, 47))
        max_imp_FCC = []
        max_imp_occ = []
        max_imp_occ_aug = []
        max_imp_sen = []

        for i, sig_ind in enumerate(signals_to_analyze):
            if self.historical:
                self.generator.load_state_dict(
                    torch.load('./ckpt/feature_%s_generator.pt' % (feature_map_mimic[sig_ind])))
            else:
                self.generator.load_state_dict(
                    torch.load('./ckpt/feature_%s_generator_nohist.pt' % (feature_map_mimic[sig_ind])))
            label, importance[i, :], mean_predicted_risk[i, :], std_predicted_risk[i, :] = self._get_feature_importance(
                signals, sig_ind=sig_ind, n_samples=10, mode='generator', learned_risk=self.learned_risk)
            _, importance_occ[i, :], _, std_predicted_risk_occ[i, :] = self._get_feature_importance(signals,
                                                                                                    sig_ind=sig_ind,
                                                                                                    n_samples=10,
                                                                                                    mode="feature_occlusion",
                                                                                                    learned_risk=self.learned_risk)
            _, importance_occ_aug[i, :], _, std_predicted_risk_occ_aug[i, :] = self._get_feature_importance(signals,
                                                                                                            sig_ind=sig_ind,
                                                                                                            n_samples=10,
                                                                                                            mode='augmented_feature_occlusion',
                                                                                                            learned_risk=self.learned_risk)
            max_imp_FCC.append((i, max(importance[i, :])))
            max_imp_occ.append((i, max(importance_occ[i, :])))
            max_imp_occ_aug.append((i, max(importance_occ_aug[i, :])))
            max_imp_sen.append((i, max(sensitivity_analysis_importance[i, :])))

            if retain_style:
                f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
                orders = np.argsort(importance, axis=0)
                # Plot the original signals
                for i, ref_ind in enumerate(signals_to_analyze):
                    c = color_map[ref_ind]
                    ax1.plot(np.array(signals[ref_ind, 1:] / max(abs(signals[ref_ind, 1:]))), linewidth=3, color=c,
                             label='%s' % (self.feature_map[ref_ind]))

                for imps in orders[-3:, :]:
                    for time in range(len(imps)):
                        imp = importance[imps[time], time]
                        texts = self.feature_map[imps[time]]
                        ax2.text(time, imp, texts)
                    ax2.set_ylim(0, np.max(importance))
                    ax2.set_xlim(0, 48)

                orders = np.argsort(importance_occ_aug, axis=0)
                for imps in orders[-3:, :]:
                    for time in range(len(imps)):
                        imp = importance_occ_aug[imps[time], time]
                        texts = self.feature_map[imps[time]]
                        ax3.text(time, imp, texts)
                    ax3.set_ylim(0, np.max(importance_occ_aug))
                    ax3.set_xlim(0, 48)

                orders = np.argsort(importance_occ, axis=0)
                for imps in orders[-3:, :]:
                    for time in range(len(imps)):
                        imp = importance_occ[imps[time], time]
                        texts = self.feature_map[imps[time]]
                        ax4.text(time, imp, texts)
                    ax4.set_ylim(0, np.max(importance_occ))
                    ax4.set_xlim(0, 48)

                ax1.set_title('Time series signals and Model\'s predicted risk', fontweight='bold', fontsize=34)
                ax2.set_title('FFC', fontweight='bold', fontsize=34)
                ax3.set_title('AFO', fontweight='bold', fontsize=34)
                ax4.set_title('Suresh et. al', fontweight='bold', fontsize=34)
                f.set_figheight(25)
                f.set_figwidth(30)
                plt.show()

            else:
                f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
                t = np.arange(47)
                ## Pick the most influential signals and plot their importance over time
                max_imp_FCC.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_occ.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_occ_aug.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_sen.sort(key=lambda pair: pair[1], reverse=True)

                n_feats_to_plot = min(self.timeseries_feature_size, n_important_features)
                if hasattr(self.patient_data, 'test_intervention'):
                    f_color = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
                    for int_ind, intervention in enumerate(self.patient_data.test_intervention[subject, :, :]):
                        if sum(intervention) != 0:
                            switch_point = []
                            intervention = intervention[1:]
                            for i in range(1, len(intervention)):
                                if intervention[i] != intervention[i - 1]:
                                    switch_point.append(i)
                            if len(switch_point) % 2 == 1:
                                switch_point.append(len(intervention) - 1)
                            for count in range(int(len(switch_point) / 2)):
                                if count == 0:
                                    ax1.axvspan(xmin=switch_point[count * 2], xmax=switch_point[2 * count + 1],
                                                facecolor=f_color[int_ind % len(f_color)], alpha=0.2,
                                                label='%s' % (intervention_list[int_ind]))
                                else:
                                    ax1.axvspan(xmin=switch_point[count * 2], xmax=switch_point[2 * count + 1],
                                                facecolor=f_color[int_ind % len(f_color)], alpha=0.2)

                markers = ['*', 'D', 'X', 'o', '8', 'v', '+']
                l_style = ['-.', '--', ':']
                important_signals = []

                # FCC
                for ind, sig in max_imp_FCC[0:n_feats_to_plot]:
                    ref_ind = signals_to_analyze[ind]
                    if ref_ind not in important_signals:
                        important_signals.append(ref_ind)
                    c = color_map[ref_ind]
                    ax2.errorbar(t, importance[ind, :], yerr=std_predicted_risk[ind, :],
                                 marker=markers[list(important_signals).index(ref_ind) % len(markers)],
                                 markersize=9, markeredgecolor='k', linewidth=3,
                                 linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)], color=c,
                                 label='%s' % (self.feature_map[ref_ind]))

                # Augmented feature occlusion
                for ind, sig in max_imp_occ_aug[0:n_feats_to_plot]:
                    ref_ind = signals_to_analyze[ind]
                    if ref_ind not in important_signals:
                        important_signals.append(ref_ind)
                    c = color_map[ref_ind]
                    ax3.errorbar(t, importance_occ_aug[ind, :], yerr=std_predicted_risk_occ_aug[ind, :],
                                 marker=markers[list(important_signals).index(ref_ind) % len(markers)], linewidth=3,
                                 linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)],
                                 markersize=9, markeredgecolor='k', color=c, label='%s' % (self.feature_map[ref_ind]))

                # Feature occlusion
                for ind, sig in max_imp_occ[0:n_feats_to_plot]:
                    ref_ind = signals_to_analyze[ind]
                    if ref_ind not in important_signals:
                        important_signals.append(ref_ind)
                    c = color_map[ref_ind]
                    ax4.errorbar(t, importance_occ[ind, :], yerr=std_predicted_risk_occ[ind, :],
                                 marker=markers[list(important_signals).index(ref_ind) % len(markers)],
                                 linewidth=3, linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)],
                                 markersize=9, markeredgecolor='k',
                                 color=c, label='%s' % (self.feature_map[ref_ind]))

                # Sensitivity analysis
                for ind, sig in max_imp_sen[0:n_feats_to_plot]:
                    ref_ind = signals_to_analyze[ind]
                    if ref_ind not in important_signals:
                        important_signals.append(ref_ind)
                    c = color_map[ref_ind]
                    ax5.plot(abs(sensitivity_analysis_importance[ind, :-1]), linewidth=3,
                             linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)],
                             color=c, label='%s' % (self.feature_map[ref_ind]))

                # Plot the original signals
                important_signals = np.unique(important_signals)
                for i, ref_ind in enumerate(important_signals):
                    c = color_map[ref_ind]
                    ax1.plot(np.array(signals[ref_ind, 1:] / max(abs(signals[ref_ind, 1:]))), linewidth=3,
                             linestyle=l_style[i % len(l_style)], color=c,
                             label='%s' % (self.feature_map[ref_ind]))
                ax1.plot(np.array(label), '-', linewidth=6, label='Risk score')
                ax1.grid()
                ax1.tick_params(axis='both', labelsize=26)
                ax2.grid()
                ax2.tick_params(axis='both', labelsize=26)
                ax3.grid()
                ax3.tick_params(axis='both', labelsize=26)
                ax4.grid()
                ax4.tick_params(axis='both', labelsize=26)
                ax5.grid()
                ax5.tick_params(axis='both', labelsize=26)
                ax1.set_title('Time series signals and Model\'s predicted risk', fontweight='bold', fontsize=34)
                ax2.set_title('FFC', fontweight='bold', fontsize=34)
                ax3.set_title('AFO', fontweight='bold', fontsize=34)
                ax4.set_title('Suresh et. al', fontweight='bold', fontsize=34)
                ax5.set_title('Sensitivity analysis', fontweight='bold', fontsize=34)
                ax5.set_xlabel('time', fontweight='bold', fontsize=24)
                ax1.set_ylabel('signal value', fontweight='bold',fontsize=18)
                ax2.set_ylabel('importance score', fontweight='bold', fontsize=18)
                ax3.set_ylabel('importance score', fontweight='bold', fontsize=18)
                ax4.set_ylabel('importance score', fontweight='bold', fontsize=18)
                ax5.set_ylabel('importance score', fontweight='bold', fontsize=18)
                f.set_figheight(25)
                f.set_figwidth(30)
                plt.subplots_adjust(hspace=0.5)
                plt.savefig('./examples/baselines/feature_%d.pdf' %(subject), dpi=300, orientation='landscape',
                            bbox_inches='tight')
                fig_legend = plt.figure(figsize=(13, 1.2))
                handles, labels = ax1.get_legend_handles_labels()
                plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
                fig_legend.savefig('./examples/baselines/legend_%d.pdf' %(subject), dpi=300, bbox_inches='tight')

    def train(self, n_epochs, n_features):
        for feature_to_predict in range(0,28):#(n_features):
            print('**** training to sample feature: ', feature_to_predict)
            if self.simulation:
                self.generator = FeatureGenerator(self.feature_size, self.historical).to(self.device)
            else:
                self.generator = FeatureGenerator(self.feature_size, self.historical, hidden_size=self.generator_hidden_size, prediction_size=self.prediction_size).to(self.device)
            train_feature_generator(self.generator, self.train_loader, self.valid_loader, feature_to_predict, n_epochs, self.historical)

    def _get_feature_importance(self, signal, sig_ind, n_samples=10, mode="feature_occlusion", learned_risk=False):
        self.generator.eval()
        feature_dist = np.sort(np.array(self.feature_dist[:,sig_ind,:]).reshape(-1))
        feature_dist_0 = (np.array(self.feature_dist_0[:, sig_ind, :]).reshape(-1))
        feature_dist_1 = (np.array(self.feature_dist_1[:, sig_ind, :]).reshape(-1))

        risks = []
        importance = []
        mean_predicted_risk = []
        std_predicted_risk = []
        for t in range(1,signal.shape[1]):
            if self.simulation:
                if not learned_risk:
                    risk = self.risk_predictor(signal.cpu().detach().numpy(), t)
                else:
                    risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t + 1))[:,t].item()
            else:
                risk = self.risk_predictor(signal[:,0:t+self.generator.prediction_size].view(1, signal.shape[0], t+self.generator.prediction_size)).item()
                # risk = self.risk_predictor(signal.view(1, signal.shape[0], signal.shape[1])).item()
            signal_known = torch.cat((signal[:sig_ind,t], signal[sig_ind+1:,t])).to(self.device)
            signal = signal.to(self.device)
            #risks.append(risk)
            # print('Predicted risk score at 19th hour (based on the generator): ', self.risk_predictor(generated_sig.to(self.device)).item())
            predicted_risks = []
            for _ in range(n_samples):
                # Replace signal with random sample from the distribution if feature_occlusion==True,
                # else use the generator model to estimate the value
                if mode=="feature_occlusion":
                    # prediction = torch.Tensor(np.array(np.random.uniform(low=-2*self.patient_data.feature_std[sig_ind,0], high=2*self.patient_data.feature_std[sig_ind,0])).reshape(-1)).to(self.device)
                    prediction = torch.Tensor(np.array([np.random.randn()]).reshape(-1)).to(self.device)
                elif mode=="augmented_feature_occlusion":
                    if self.risk_predictor(signal[:,0:t].view(1, signal.shape[0], t)).item() > 0.5:
                        prediction = torch.Tensor(np.array(np.random.choice(feature_dist_0)).reshape(-1,)).to(self.device)
                    else:
                        prediction = torch.Tensor(np.array(np.random.choice(feature_dist_1)).reshape(-1,)).to(self.device)
                elif mode=="generator" or mode=="combined":
                    prediction, _ = self.generator(signal_known.view(1,-1), signal[:, 0:t].view(1,signal.size(0),t))
                    if mode=="combined":
                        if self.risk_predictor(signal[:,0:t].view(1, signal.shape[0], t)).item() > 0.5:
                            prediction = torch.Tensor(self._find_closest(feature_dist_0, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)
                        else:
                            prediction = torch.Tensor(self._find_closest(feature_dist_1, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)
                        # prediction = torch.Tensor(self._find_closest(feature_dist, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)
                predicted_signal = signal[:,0:t+self.generator.prediction_size].clone()
                # predicted_signal = signal[:, :].clone()
                predicted_signal[:,t:t+self.generator.prediction_size] = torch.cat((signal[:sig_ind,t:t+self.generator.prediction_size], prediction.view(1,-1), signal[sig_ind+1:,t:t+self.generator.prediction_size]),0)
                if self.simulation:
                    if not learned_risk:
                        predicted_risk = self.risk_predictor(predicted_signal.cpu().detach().numpy(), t)
                    else:
                        predicted_risk = self.risk_predictor(predicted_signal[:,0:t].view(1,predicted_signal.shape[0],t+self.generator.prediction_size+1).to(self.device))[:,t+self.generator.prediction_size].item()
                else:
                    predicted_risk = self.risk_predictor(predicted_signal[:, 0:t + self.generator.prediction_size].view(1, predicted_signal.shape[0], t + self.generator.prediction_size).to(self.device)).item()
                    # predicted_risk = self.risk_predictor(predicted_signal.view(1, predicted_signal.shape[0], predicted_signal.shape[1]).to(self.device)).item()
                predicted_risks.append(predicted_risk)
            risks.append(risk)
            predicted_risks = np.array(predicted_risks)
            mean_imp = np.mean(predicted_risks,0)
            std_imp = np.std(predicted_risks, 0)
            mean_predicted_risk.append(mean_imp)
            std_predicted_risk.append(std_imp)
            importance.append(abs(mean_imp-risk))
        return risks, importance, mean_predicted_risk, std_predicted_risk

    def get_stats(self):
        tp = [[]]*8
        fn = [[]]*8
        tn = [[]]*8
        fp = [[]]*8
        for i,sig in enumerate(range(20,28)):
            self.generator.load_state_dict(torch.load('./ckpt/feature_%d_generator.pt'%(sig)))
            for signals,label in list(self.test_loader.dataset):
                pred, importance, mean_predicted_risk, std_predicted_risk = self._get_feature_importance(signals,
                                                                                                          sig_ind=sig)
                imp = np.mean(mean_predicted_risk)
                if label==1:
                    if pred[-1]>.5:
                        # True positive
                        tp[i].append(imp)
                    if pred[-1] <= .5:
                        # False negative
                        fn[i].append(imp)
                if label==0:
                    if pred[-1] >.5:
                        # False positive
                        fp[i].append(imp)
                    if pred[-1] <= .5:
                        # True negative
                        tn[i].append(imp)
        tp = [np.mean(imps) for imps in tp]
        fn = [np.mean(imps) for imps in fn]
        tn = [np.mean(imps) for imps in tn]
        fp = [np.mean(imps) for imps in fp]
        print("Importance of TRUE POSITIVES: ", tp)
        print("Importance of FALSE NEGATIVES: ", fn)
        print("Importance of TRUE NEGATIVES: ", tn)
        print("Importance of FALSE POSITIVES: ", fp)

    def summary_stat(self, intervention_ID=11):
        testset = list(self.test_loader.dataset)
        signals = torch.stack(([x[0] for x in testset])).to(self.device)
        labels = torch.stack(([x[1] for x in testset])).to(self.device)
        interventions = self.patient_data.test_intervention[:,intervention_ID,:]
        df = pd.DataFrame(columns = ['pid','intervention_id','method','top1','top2','top3'])
        if hasattr(self.patient_data, 'test_intervention'):
            ind_list = np.where(np.sum(interventions[:,1:],axis=1)!=0)[0] ## Index of subject that have intervention=intervention_ID data recorded

            ## Sensitivity analysis
            test_signals = torch.stack([testset[sample][0] for sample in ind_list])
            sensitivity_analysis = np.zeros((test_signals.shape))
            # self.risk_predictor = LR(self.input_size).to(self.device)
            # self.risk_predictor.load_state_dict(torch.load('./ckpt/LR.pt'))
            self.risk_predictor.train()
            for t in range(1,test_signals.size(2)):
                signal_t = torch.Tensor(test_signals[:,:,:t+1]).to(self.device).requires_grad_()
                out = self.risk_predictor(signal_t)
                for s in range(len(ind_list)):
                    out[s, 0].backward(retain_graph=True)
                    sensitivity_analysis[s,:,t] = signal_t.grad.data[s,:,t].cpu().detach().numpy()
                    signal_t.grad.data.zero_()
            self.risk_predictor.eval()

            # self.risk_predictor = EncoderRNN(self.input_size, hidden_size=150, rnn='GRU', regres=True).to(self.device)
            # self.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
            for ind, subject in enumerate(ind_list):
                label = labels[subject]
                signal = signals[subject,:,:]
                intervention = interventions[subject, 1:]
                start_point = np.argwhere(intervention==1)[0][0]

                # print(start_point)
                if start_point<10:
                    continue
                signals_analyze = range(27)
                max_imp_FCC = []
                max_imp_occ = []
                max_imp_sen = []
                importance = np.zeros((len(signals_analyze),start_point))
                importance_occ = np.zeros((len(signals_analyze), start_point))

                self.risk_predictor.eval()
                prediction = int(self.risk_predictor(signal[:,:start_point+1].view(1,signal.shape[0],start_point+1)).item()>0.5)
                if prediction<0.7 or prediction!=label:
                    continue
                for i in range(27):
                    label, importance[i, :], _, _ = self._get_feature_importance(signal[:,:start_point+1], sig_ind=i, n_samples=10,
                                                                                mode='combined', learned_risk=self.learned_risk)
                    _, importance_occ[i, :], _, _ = self._get_feature_importance(signal[:,:start_point+1], sig_ind=i, n_samples=10,
                                                                                 mode="feature_occlusion", learned_risk=self.learned_risk)
                    max_imp_FCC.append((i, max(importance[i, :])))
                    max_imp_occ.append((i, max(importance_occ[i, :])))
                    max_imp_sen.append((i, sensitivity_analysis[ind,i,start_point+1]))
                max_imp_FCC.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_occ.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_sen.sort(key=lambda pair: pair[1], reverse=True)

                # print('************ Top 5 signals:')
                # print('FCC: ', max_imp_FCC[0:5])
                # print('Feature occlusion: ', max_imp_occ[0:5])
                # print('Sensitivity analysis: ', max_imp_sen[0:5])
                df.loc[-1] = [subject,intervention_ID,'FCC',max_imp_FCC[0][0],max_imp_FCC[1][0],max_imp_FCC[2][0]]  # adding a row
                df.index = df.index + 1
                df.loc[-1] = [subject,intervention_ID,'f_occ',max_imp_occ[0][0],max_imp_occ[1][0],max_imp_occ[2][0]]  # adding a row
                df.index = df.index + 1
                df.loc[-1] = [subject,intervention_ID,'sensitivity',max_imp_sen[0][0],max_imp_sen[1][0],max_imp_sen[2][0]]  # adding a row
                df.index = df.index + 1
                ## Plot intervention
                # switch_point = []
                # if intervention[0]==1:
                #     switch_point.append(0)
                # for i in range(1, len(intervention)):
                #     if intervention[i] != intervention[i - 1]:
                #         switch_point.append(i)
                # if len(switch_point) % 2 == 1:
                #     switch_point.append(len(intervention) - 1)
                # for count in range(int(len(switch_point) / 2)):
                #     if count == 0:
                #         plt.axvspan(xmin=switch_point[count * 2], xmax=switch_point[2 * count + 1], alpha=0.2)
                #     else:
                #         plt.axvspan(xmin=switch_point[count * 2], xmax=switch_point[2 * count + 1], alpha=0.2)
                # plt.plot(signals[subject,0,1:])
                # plt.show()
            print(df)
            df.to_pickle("./interventions/int_%d.pkl"%(intervention_ID))

    def plot_summary_stat(self, intervention_ID=1):
        df = pd.read_pickle("./interventions/int_%d.pkl" % (intervention_ID))
        fcc_df = df.loc[df['method']=='FCC']
        occ_df = df.loc[df['method'] == 'f_occ']
        sen_df = df.loc[df['method'] == 'sensitivity']
        fcc_dist = np.sort(np.array(fcc_df[['top1','top2','top3']]).reshape(-1,))
        occ_dist = np.sort(np.array(occ_df[['top1', 'top2', 'top3']]).reshape(-1, ))
        sen_dist = np.sort(np.array(sen_df[['top1', 'top2', 'top3']]).reshape(-1, ))


        # color_map = plt.get_cmap("tab20")(np.linspace(0, 1, 28))
        fcc_top = self._create_pairs(self._find_count(fcc_dist))[0:6]
        occ_top = self._create_pairs(self._find_count(occ_dist))[0:6]
        sen_top = self._create_pairs(self._find_count(sen_dist))[0:6]
        f, (ax1, ax2, ax3) = plt.subplots(3,1, sharey=True)
        ax1.bar([self.feature_map[x[0]] for x in fcc_top], [x[1] for x in fcc_top], color=[color_map[x[0]] for x in fcc_top])
        ax2.bar([self.feature_map[x[0]] for x in occ_top], [x[1] for x in occ_top], color=[color_map[x[0]] for x in occ_top])
        ax3.bar([self.feature_map[x[0]] for x in sen_top], [x[1] for x in sen_top], color=[color_map[x[0]] for x in sen_top])
        f.suptitle('%s'%(intervention_list[intervention_ID]), fontweight='bold', fontsize=28)
        ax1.set_title('FFC', fontsize=24, fontweight='bold')
        ax2.set_title('Suresh et. al', fontsize=24, fontweight='bold')
        ax3.set_title('Sensitivity analysis', fontsize=24, fontweight='bold')
        ax1.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        ax3.tick_params(labelsize=20)
        plt.subplots_adjust(hspace=0.3)
        f.set_figheight(12)
        f.set_figwidth(15)#(10)
        plt.savefig('./examples/distributions/top_%s'%(intervention_list[intervention_ID]), dpi=300, bbox_inches='tight')


        # for rank in range(3):
        #     ind = argmax(fcc_dist)
        #     fcc_x[rank] = intervention_list[ind]
        #     fcc_y = fcc_dist[ind]
        #     fcc_
        f, (ax1,ax2,ax3) = plt.subplots(3, sharex=True)
        ax1.bar(self.feature_map, self._find_count(fcc_dist))
        ax2.bar(self.feature_map, self._find_count(occ_dist))
        ax3.bar(self.feature_map, self._find_count(sen_dist))
        ax1.set_title('FFC importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        ax2.set_title('feature occlusion importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        ax3.set_title('sensitivity analysis importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        f.set_figheight(10)
        f.set_figwidth(20)
        plt.savefig('./examples/distributions/%s'%(intervention_list[intervention_ID]))
        # plt.show()

    def _create_pairs(self, a):
        l=[]
        for i,element in enumerate(a):
            l.append((i,element))
        l.sort(key=lambda x: x[1], reverse=True)
        return l

    def _find_count(self, a):
        count_arr = np.zeros(len(self.feature_map),)
        for elem in a:
            count_arr[elem] += 1
        return count_arr

    def _find_closest(self, arr, target):
        n = len(arr)
        # Corner cases
        if (target <= arr[0]):
            return arr[0]
        if (target >= arr[n - 1]):
            return arr[n - 1]

        # Doing binary search
        i = 0;
        j = n;
        mid = 0
        while (i < j):
            mid = (i + j) // 2
            if (arr[mid] == target):
                return arr[mid]
            # If target is less than array
            # element, then search in left
            if (target < arr[mid]):
                # If target is greater than previous
                # to mid, return closest of two
                if (mid > 0 and target > arr[mid - 1]):
                    return self._get_closest(arr[mid - 1], arr[mid], target)

                    # Repeat for left half
                j = mid

                # If target is greater than mid
            else:
                if (mid < n - 1 and target < arr[mid + 1]):
                    return self._get_closest(arr[mid], arr[mid + 1], target)

                    # update i
                i = mid + 1

        # Only single element left after search
        return arr[mid]

    def _get_closest(self,val1, val2, target):

        if (target - val1 >= val2 - target):
            return val2
        else:
            return val1


