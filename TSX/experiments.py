import torch
import os
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, train_model_rt, test, test_rt, logistic
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE
from TSX.generator import Generator, test_generator, FeatureGenerator, test_feature_generator, train_feature_generator
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

feature_map_simulation = ['var 0', 'var 1', 'var 2']

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
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor',simulation=False):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader)
        #self.state_encoder = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=False)
        #self.risk_predictor = RiskPredictor(encoding_size)
        #self.model = torch.nn.Sequential(self.state_encoder, self.risk_predictor)
        if simulation:
            self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=True)
        else:
            self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=False)
        self.experiment = experiment
        self.simulation = simulation

    def run(self, train):
        if train:
            self.train(n_epochs=120,learn_rt=self.simulation)
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
    """ Generating feature importance over time using a generative model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, historical=False, simulation=False,  experiment='feature_generator_explainer'):
        super(FeatureGeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader)
        self.generator = FeatureGenerator(feature_size, historical).to(self.device)
        if not simulation:
            self.feature_size = feature_size - 4
        else:
            self.feature_size = feature_size
        self.input_size = feature_size
        self.experiment = experiment
        self.historical = historical
        self.simulation = simulation
        #this is used to see fhe difference between true risk vs learned risk for simulations
        self.learned_risk = True
        trainset = list(self.train_loader.dataset)
        self.feature_dist = torch.stack([x[0] for x in trainset])
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
            self.risk_predictor = EncoderRNN(self.input_size, hidden_size=100, rnn='GRU', regres=True)
            self.feature_map = feature_map_mimic
        self.risk_predictor = self.risk_predictor.to(self.device)

    def run(self, train):
        if train:
            self.train(n_features=self.feature_size, n_epochs=80)
        else:
            if os.path.exists('./ckpt/feature_0_generator.pt'):
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
            else:
                raise RuntimeError('No saved checkpoint for this model')

            testset = list(self.test_loader.dataset)

            if self.simulation:
                with open(os.path.join('./data_generator/data/simulated_data/thresholds_test.pkl'), 'rb') as f:
                    th = pkl.load(f)
                #For simulated data this is the last entry - end of 48 hours that's the actual outcome
                label = np.array([x[1][-1] for x in testset])
                #print(label)
                high_risk = np.where(label>=0.5)[0]
                samples_to_analyse = np.random.choice(high_risk, 5, replace=False)
            else:
                label = np.array([x[1] for x in testset])
                # high_risk = np.where(label==1)[0]
                # sub = np.random.choice(high_risk, 10, replace=False)
                samples_to_analyse = [3460,3048,3460,881,188,3845,454]#,58,218,86,45]


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
                    #print('gradient:, ', i ,signal.grad.data[i,:,:].norm(2).cpu().detach().numpy(),signal.grad.shape)
                    grad_out = np.array(grad_out)
                    sensitivity_analysis = grad_out
                    self.risk_predictor.eval()

            print('\n********** Visualizing a few samples **********')
            for sub_ind, subject in enumerate(samples_to_analyse):#range(30):
                f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
                if self.simulation:
                    signals, label_o = testset[subject]
                    label_o = label_o[-1]
                    print('Subject ID: ', subject)
                    print('Did this patient die? ', {1:'yes',0:'no'}[label_o.item()>0.5])
                else:
                    signals, label_o = testset[subject]
                    risk = []
                    for ttt in range(1,48):
                        risk.append(self.risk_predictor(signals[:, 0:ttt].view(1, signals.shape[0], ttt).to(self.device)).item())
                    print((max(risk) - min(risk)))
                    print('Subject ID: ', subject)
                    print('Did this patient die? ', {1:'yes',0:'no'}[label_o.item()])

                importance = np.zeros((self.feature_size,47))
                mean_predicted_risk = np.zeros((self.feature_size,47))
                std_predicted_risk = np.zeros((self.feature_size,47))
                importance_occ = np.zeros((self.feature_size,47))
                std_predicted_risk_occ = np.zeros((self.feature_size,47))
                importance_comb = np.zeros((self.feature_size,47))
                std_predicted_risk_comb = np.zeros((self.feature_size,47))
                f_imp = np.zeros(self.feature_size)
                f_imp_occ = np.zeros(self.feature_size)
                f_imp_comb = np.zeros(self.feature_size)
                max_imp_total = []

                for i, sig_ind in enumerate(range(0,self.feature_size)):
                    if self.historical:
                        self.generator.load_state_dict(torch.load('./ckpt/feature_%d_generator.pt'%(sig_ind)))
                    else:
                        self.generator.load_state_dict(torch.load('./ckpt/feature_%d_generator_nohist.pt'%(sig_ind)))

                    label, importance[i,:], mean_predicted_risk[i,:], std_predicted_risk[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode='generator', learned_risk=self.learned_risk)
                    _, importance_occ[i, :], _, std_predicted_risk_occ[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode="feature_occlusion",learned_risk=self.learned_risk)
                    _, importance_comb[i, :], _, std_predicted_risk_comb[i,:] = self._get_feature_importance(signals,
                                                                                                          sig_ind=sig_ind, n_samples=10, mode='combined',learned_risk=self.learned_risk)
                    max_imp_total.append((i,max(mean_predicted_risk[i,:])))

                retain_style=False
                t = np.arange(47)
                if retain_style:
                    orders = np.argsort(importance, axis=0)
                    for imps in orders[-3:,:]:
                        for time in range(len(imps)):
                            imp = importance[imps[time],time]
                            texts = self.feature_map[imps[time]]
                            ax2.text(time, imp, texts)
                        ax2.set_ylim(0,np.max(importance))
                        ax2.set_xlim(0,48)

                    orders = np.argsort(importance_occ, axis=0)
                    for imps in orders[-3:,:]:
                        for time in range(len(imps)):
                            imp = importance_occ[imps[time],time]
                            texts = self.feature_map[imps[time]]
                            ax3.text(time, imp, texts)
                        ax3.set_ylim(0,np.max(importance_occ))
                        ax3.set_xlim(0,48)

                    orders = np.argsort(importance_comb, axis=0)
                    for imps in orders[-3:,:]:
                        for time in range(len(imps)):
                            imp = importance_comb[imps[time],time]
                            texts = self.feature_map[imps[time]]
                            ax4.text(time, imp, texts)
                        ax4.set_ylim(0,np.max(importance_comb))
                        ax4.set_xlim(0,48)

                else:
                    max_imp = np.argmax(importance,axis=0)
                    for im in max_imp:
                        f_imp[im] += 1
                    max_imp_occ = np.argmax(importance_occ,axis=0)
                    for im in max_imp_occ:
                        f_imp_occ[im] += 1
                    max_imp_comb = np.argmax(importance_comb,axis=0)
                    for im in max_imp_comb:
                        f_imp_comb[im] += 1

                    ## Pick the most influential signals and plot their importance over time
                    max_imp_total.sort(key=lambda pair: pair[1], reverse=True)

                    n_feats_to_plot = min(self.feature_size,4)
                    for ind,sig in max_imp_total[0:n_feats_to_plot]:
                    # for ind in np.argsort(f_imp)[-4:]:# range(4):
                        #ax1.plot(np.array(signals[ind,:]), label='%s'%(feature_map[ind]))
                        ax2.errorbar(t, importance[ind,:], yerr=std_predicted_risk[ind,:], marker='^', label='%s importance'%(self.feature_map[ind]))
                        # plt.plot(abs(signal.grad.data[sub_ind,i,:].cpu().detach().numpy()*1e+2), label='Feature %s Sensitivity analysis'%(feature_map[sig_ind]))
                    for ind, sig in max_imp_total[0:n_feats_to_plot]:
                    #for ind in np.argsort(f_imp_occ)[-4:]:# range(4):
                        ax3.errorbar(t, importance_occ[ind, :], yerr=std_predicted_risk_occ[ind, :], marker='^',
                                     label='%s importance' % (self.feature_map[ind]))
                        #ax1.plot(np.array(signals[ind,:]), label='%s'%(feature_map[ind]))
                    for ind, sig in max_imp_total[0:n_feats_to_plot]:
                    #for ind in np.argsort(f_imp_comb)[-4:]:# range(4):
                        ax1.plot(np.array(signals[ind,:]), label='%s'%(self.feature_map[ind]))
                        ax4.errorbar(t, importance_comb[ind, :], yerr=std_predicted_risk_comb[ind, :], marker='^', label='%s importance' % (self.feature_map[ind]))

                    for ind,sig in max_imp_total[0:n_feats_to_plot]:
                        ax5.plot(t,abs(sensitivity_analysis[sub_ind,ind,:-1]),label='%s'%(self.feature_map[ind]))
                ax1.plot(t, np.array(label), '--', label='Risk score')
                ax1.legend()
                ax2.legend()
                ax3.legend()
                ax4.legend()
                ax5.legend()
                ax1.grid()
                ax2.grid()
                ax3.grid()
                ax4.grid()
                ax5.grid()
                ax1.set_title('Time series signals')
                ax2.set_title('Signal importance using generative model')
                ax3.set_title('Signal importance using feature occlusion')
                ax4.set_title('Signal importance using combined methods')
                ax5.set_title('Signal importance using sensitivity analysis')
                plt.show()

    def train(self, n_epochs, n_features):
        for feature_to_predict in range(n_features):
            print('**** training to sample feature: ', feature_to_predict)
            train_feature_generator(self.generator, self.train_loader, self.valid_loader, feature_to_predict, n_epochs, self.historical)

    def _get_feature_importance(self, signal, sig_ind, n_samples=10, mode="feature_occlusion",learned_risk=False):
        self.generator.eval()
        feature_dist = np.sort(np.array(self.feature_dist[:,sig_ind,:]).reshape(-1))

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
                risk = self.risk_predictor(signal[:,0:t + 1].view(1, signal.shape[0], t + 1)).item()
                # risk = self.risk_predictor(signal.view(1, signal.shape[0], signal.shape[1])).item()
            signal_known = torch.cat((signal[:sig_ind,t], signal[sig_ind+1:,t])).to(self.device)
            signal = signal.to(self.device)
            risks.append(risk)
            # print('Predicted risk score at 19th hour (based on the generator): ', self.risk_predictor(generated_sig.to(self.device)).item())
            predicted_risks = []
            for _ in range(n_samples):
                # Replace signal with random sample from the distribution if feature_occlusion==True,
                # else use the generator model to estimate the value
                if mode=="feature_occlusion":
                    prediction = torch.Tensor(np.random.choice(feature_dist).reshape(-1,)).to(self.device)
                elif mode=="generator":
                    prediction, _ = self.generator(signal_known.view(1,-1), signal[:, 0:t].view(1,signal.size(0),t))
                elif mode=="combined":
                    prediction1, _ = self.generator(signal_known.view(1, -1), signal[:, 0:t].view(1, signal.size(0), t))
                    prediction = torch.Tensor( self._find_closest(feature_dist, prediction1.cpu().detach().numpy()).reshape(-1)).to(self.device)
                predicted_signal = signal[:,0:t+1].clone()
                # predicted_signal = signal[:, :].clone()
                predicted_signal[:,t] = torch.cat((signal[:sig_ind,t], prediction.view(-1), signal[sig_ind+1:,t]),0)
                if self.simulation:
                    if not learned_risk:
                        predicted_risk = self.risk_predictor(predicted_signal.cpu().detach().numpy(), t)
                    else:
                        predicted_risk = self.risk_predictor(predicted_signal[:,0:t+1].view(1,predicted_signal.shape[0],t+1).to(self.device))[:,t].item()
                else:
                    predicted_risk = self.risk_predictor(predicted_signal[:, 0:t + 1].view(1, predicted_signal.shape[0], t + 1).to(self.device)).item()
                    # predicted_risk = self.risk_predictor(predicted_signal.view(1, predicted_signal.shape[0], predicted_signal.shape[1]).to(self.device)).item()
                predicted_risks.append(predicted_risk)
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

