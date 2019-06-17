import torch
import os
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, train_model_rt, train_model_rt_rg, test, test_model_rt, test_model_rt_rg, logistic
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE
from TSX.generator import Generator, test_generator, FeatureGenerator, test_feature_generator, train_feature_generator
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pickle as pkl
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score

feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

feature_map_simulation = ['var 0', 'var 1', 'var 2']
simulation_color_map = ['#3cb44b', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#000075', '#a9a9a9', '#ffffff', '#000000','#ffe119','#e6194B']
simulation_color_map = ['#e6194B', '#469990', '#000000','#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',  '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#3cb44b','#ffe119']
feature_map_ghg = [str(i+1) for i in range(15)]
scatter_map_ghg={}
for i in range(15):
    scatter_map_ghg[feature_map_ghg[i]]=[]
line_styles_map=['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--','-.',':']
marker_styles_map=['o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h']

width=500
height=560
scatter_map_ghg['1'] = [174,height-60]
scatter_map_ghg['2'] = [59,height-81]
scatter_map_ghg['3'] = [126,height-100]
scatter_map_ghg['4'] = [181,height-161]
scatter_map_ghg['5'] = [101,height-200]
scatter_map_ghg['6'] = [294,height-289]
scatter_map_ghg['7'] = [106,height-226]
scatter_map_ghg['8'] = [178,height-291]
scatter_map_ghg['9'] = [141,height-315]
scatter_map_ghg['10'] = [405,height-420]
scatter_map_ghg['11'] = [190,height-388]
scatter_map_ghg['12'] = [291,height-453]
scatter_map_ghg['13'] = [385,height-489]
scatter_map_ghg['14'] = [383,height-516]
scatter_map_ghg['15'] = [310,height-212]


class Experiment(ABC):
    def __init__(self, train_loader, valid_loader, test_loader,data='mimic'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.data=data

    @abstractmethod
    def run(self):
        raise RuntimeError('Function not implemented')

    def train(self, n_epochs,learn_rt=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        
        if not learn_rt:
            train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment,data=self.data)
            # Evaluate performance on held-out test set
            _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            print('\nFinal performance on held out test set ===> AUC: ', auc_test)
        else:
            if self.data=='mimic':
                train_model_rt(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment,data=self.data)
            elif self.data=='ghg' or self.data=='simulation':
                train_model_rt_rg(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment,data=self.data)

class Baseline(Experiment):
    """ Baseline mortality prediction using a logistic regressions model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, experiment='baseline',data='mimic'):
        super(Baseline, self).__init__(train_loader, valid_loader, test_loader, data=data)
        self.model = LR(feature_size).to(self.device)
        self.experiment = experiment
        self.data=data

    def run(self, train):
        if train:
            self.train(n_epochs=120)
        else:
            if os.path.exists('./ckpt/' + self.data + '/' + str(self.experiment) + '.pt'):
                self.model.load_state_dict(torch.load('./ckpt/' + self.data + '/'+ str(self.experiment) + '.pt'))
                _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                print('Loading model with AUC: ', auc_test)
            else:
                raise RuntimeError('No saved checkpoint for this model')


class EncoderPredictor(Experiment):
    """ Baseline mortality prediction using an encoder to encode patient status, and a risk predictor to predict risk of mortality
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor',simulation=False,data='mimic'):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader,data=data)
        #self.state_encoder = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=False)
        #self.risk_predictor = RiskPredictor(encoding_size)
        #self.model = torch.nn.Sequential(self.state_encoder, self.risk_predictor)
        if simulation:
            self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=False,data='simulation')
        elif data=='mimic' or data=='ghg':
            self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=False,data=data)

        self.experiment = experiment
        self.simulation = simulation
        self.data=data

    def run(self, train,n_epochs=60,loader_obj=None):
        if train:
            self.train(n_epochs=n_epochs,learn_rt=self.data=='ghg')
        else:
            if os.path.exists(os.path.join('./ckpt/',self.data, str(self.experiment) + '.pt')):
                self.model.load_state_dict(torch.load(os.path.join('./ckpt/' ,self.data, str(self.experiment) + '.pt')))

                if not self.data=='ghg':
                    _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                else:
                    test_loss = test_model_rt(self.test_loader, self.model, self.device)
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
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, historical=False, simulation=False,  experiment='feature_generator_explainer',data='mimic'):
        super(FeatureGeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader)
        if simulation:
            self.generator = FeatureGenerator(feature_size, historical, hidden_size=10, data=data).to(self.device)
        else:
            self.generator = FeatureGenerator(feature_size, historical, data=data).to(self.device)

        if not simulation and data=='mimic':
            self.feature_size = feature_size - 4
        else:
            self.feature_size = feature_size
        self.input_size = feature_size
        self.experiment = experiment
        self.historical = historical
        self.simulation = simulation

        if self.simulation:
            self.data = 'simulation'
        else:
            self.data = data

        print('setting up generator for ', self.data)

        #this is used to see fhe difference between true risk vs learned risk for simulations
        self.learned_risk = True
        trainset = list(self.train_loader.dataset)
        self.feature_dist = torch.stack([x[0] for x in trainset])
        if simulation:
            if self.simulation and not self.learned_risk:
                #self.risk_predictor = lambda signal,ph,t:logistic(2.5*((ph[t]==0)*signal[0, t] * signal[0, t] + (ph[t]==1)*signal[1,t] * signal[1,t] + (ph[t]==2)*signal[2, t] * signal[2, t]) - 1)
                self.risk_predictor = lambda signal,ph,t:logistic(2.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t]) - 1)
            #elif self.simulation and self.learned_risk:
            #    self.risk_predictor = EncoderRNN(feature_size,hidden_size=20,rnn='GRU',regres=True, return_all=False)
            else:
                self.risk_predictor = EncoderRNN(feature_size,hidden_size=5,rnn='GRU',regres=True, return_all=False,data=data)
                self.risk_predictor = self.risk_predictor.to(self.device)

            self.feature_map = feature_map_simulation
        else:
            if data=='mimic':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=100, rnn='GRU', regres=True,data=data)
                self.feature_map = feature_map_mimic
                self.risk_predictor = self.risk_predictor.to(self.device)
            elif data=='ghg':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=100, rnn='GRU', regres=True,data=data)
                self.feature_map = feature_map_ghg
                self.risk_predictor = self.risk_predictor.to(self.device)

    def run(self, train,n_epochs=80,loader_obj=None):
        if train:
            self.train(n_features=self.feature_size, n_epochs=n_epochs)
        else:
            ckpt_path = os.path.join('./ckpt',self.data)
            if self.historical:
                check_path = os.path.join(ckpt_path,'feature_0_generator.pt')
            else:
                check_path = os.path.join(ckpt_path,'feature_0_generator_nohist.pt')

            if os.path.exists(check_path):
                if not self.simulation:
                    self.risk_predictor.load_state_dict(torch.load(os.path.join(ckpt_path,'risk_predictor.pt')))
                    self.risk_predictor = self.risk_predictor.to(self.device)
                    self.risk_predictor.eval()
                else: #simulated data
                    if self.learned_risk:
                        print('loading risk predictor from path:', ckpt_path)
                        self.risk_predictor.load_state_dict(torch.load(os.path.join(ckpt_path,'risk_predictor.pt')))
                        self.risk_predictor = self.risk_predictor.to(self.device)
                        self.risk_predictor.eval()
                #gen_test_loss = test_feature_generator(self.generator, self.test_loader, 1)
                #print('Generator test loss: ', gen_test_loss)
            else:
                raise RuntimeError('No saved checkpoint for this model')

            testset = list(self.test_loader.dataset)

            if self.simulation:
                # Load ground truth feature importance
                with open(os.path.join('./data_generator/data/simulated_data/thresholds_test.pkl'), 'rb') as f:
                    th = pkl.load(f)
 
                with open(os.path.join('./data_generator/data/simulated_data/gt_test.pkl'), 'rb') as f:
                    gt_importance = pkl.load(f)
                    #gt_importance = gt_importance[:,:,0]

                #For simulated data this is the last entry - end of 48 hours that's the actual outcome
                label = np.array([x[1] for x in testset])
                #print(label)
                #high_risk = np.arange(label.shape[0])
                high_risk = np.where(label==1)[0]
                samples_to_analyse = np.random.choice(high_risk, 10, replace=False)
            else:
                if self.data=='mimic':
                    label = np.array([x[1] for x in testset])
                    # high_risk = np.where(label==1)[0]
                    # sub = np.random.choice(high_risk, 10, replace=False)
                    samples_to_analyse = [3460,3048,3460,881,188,3845,454]#,58,218,86,45]
                elif self.data=='ghg':
                    label = np.array([x[1][-1] for x in testset])
                    high_risk = np.arange(label.shape[0])
                    samples_to_analyse = np.random.choice(high_risk, len(high_risk), replace=False)

            ## Sensitivity analysis as a baseline
            signal = torch.stack([testset[sample][0] for sample in samples_to_analyse])

            if self.data=='ghg':
                label_tch = torch.stack([testset[sample][1] for sample in samples_to_analyse])
                signal_scaled = loader_obj.scaler_x.inverse_transform(np.reshape(signal.cpu().detach().numpy(),[len(samples_to_analyse),-1]))
                signal_scaled = np.reshape(signal_scaled,signal.shape)
                label_scaled = loader_obj.scaler_y.inverse_transform(np.reshape(label_tch.cpu().detach().numpy(),[len(samples_to_analyse),-1]))
                label_scaled = np.reshape(label_scaled,label_tch.shape)
                #label_scaled = label_tch.cpu().detach().numpy()

                tvec = [int(x) for x in np.linspace(1,signal.shape[2]-1,5)]
            else:
                tvec = list(range(1,signal.shape[2]))
                signal_scaled = signal

            nt = len(tvec)
            #print(signal.shape)
            sensitivity_analysis = np.zeros((signal.shape[0],signal.shape[1],nt))
            if not self.simulation:
                if self.data=='mimic':
                    self.risk_predictor.train()
                    #for t in range(1,signal.size(2)):
                    for t_ind,t in enumerate(tvec):
                        signal_t = torch.Tensor(signal[:,:,:t]).to(self.device).requires_grad_()
                        out = self.risk_predictor(signal_t)
                        for s in range(len(samples_to_analyse)):
                            out[s, 0].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:].cpu().detach().numpy().reshape([-1])
                            signal_t.grad.data.zero_()
                    self.risk_predictor.eval()
                elif self.data=='ghg':
                    self.risk_predictor.train()
                    for t_ind,t in enumerate(tvec):
                        signal_t = torch.Tensor(signal[:,:,:t]).to(self.device).requires_grad_()
                        out = self.risk_predictor(signal_t)
                        for s in range(len(samples_to_analyse)):
                            out[s, 0].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:,t-1].cpu().detach().numpy().reshape([-1])
                            signal_t.grad.data.zero_()
                    self.risk_predictor.eval()
                    print('sensitvity done')
            else:
                #print(testset[0][0].shape)
                if not self.learned_risk:
                    #out = np.array([np.array([self.risk_predictor(sample[0].cpu().detach().numpy(),gt_importance[i,:],t) for t in range(48)]) for i,sample in enumerate(testset) if i in samples_to_analyse])
                    grad_out = []
                    for kk,i in enumerate(samples_to_analyse):
                        sample = testset[i][0].cpu().detach().numpy()
                        gt_imp = gt_importance[i,:]
                        out = np.array([self.risk_predictor(sample,gt_imp,tt) for tt in tvec])
                        #print(out.shape, sample.shape)
                        grad_x0 = np.array([5*out[tt]*(1-out[tt])*(gt_importance[i,tt]==0)*sample[0,tt] for tt in tvec])
                        grad_x1 = np.array([5*out[tt]*(1-out[tt])*(gt_importance[i,tt]==1)*sample[1,tt] for tt in tvec])
                        grad_x2 = np.array([5*out[tt]*(1-out[tt])*(gt_importance[i,tt]==2)*sample[2,tt] for tt in tvec])
                        grad_out.append(np.stack([grad_x0, grad_x1, grad_x2]))
                    sensitivity_analysis = np.array(grad_out)
                else:
                    #In simulation data also get sensitivity w.r.t. a learned predictor
                    self.risk_predictor.train()
                    for t_ind, t in enumerate(tvec):
                        #print(t)
                        signal_t = torch.Tensor(signal[:,:,:t]).to(self.device).requires_grad_()
                        out = self.risk_predictor(signal_t)
                        for s in range(len(samples_to_analyse)):
                            out[s, 0].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:].cpu().detach().numpy()[:,0]
                            signal_t.grad.data.zero_()
                    self.risk_predictor.eval()

            print('\n********** Visualizing a few samples **********')

            imp_acc = 0
            imp_acc_occ = 0
            imp_acc_su = 0
            imp_acc_comb = 0
            imp_acc_sens = 0

            mse_vec=[]
            mse_vec_occ=[]
            mse_vec_su=[]
            mse_vec_comb=[]
            mse_vec_sens=[]

            for sub_ind, subject in enumerate(samples_to_analyse):#range(30):
                if self.data=='simulation':
                    f, (ax1,ax2, ax3, ax4) = plt.subplots(4,sharex=True,figsize=(12,10))
                else:
                    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True,figsize=(10,6))
                if self.simulation:
                    signals, label_o = testset[subject]
                    print('Subject ID: ', subject)
                    print('Did this patient die? ', {1:'yes',0:'no'}[label_o.item()>0.5])
                elif self.data=='mimic':
                    signals, label_o = testset[subject]
                    risk = []
                    for ttt in range(1,48):
                        risk.append(self.risk_predictor(signals[:, 0:ttt].view(1, signals.shape[0], ttt).to(self.device)).item())
                    print((max(risk) - min(risk)))
                    print('Subject ID: ', subject)
                    print('Did this patient die? ', {1:'yes',0:'no'}[label_o.item()])
                elif self.data=='ghg':
                    signals, label_o = testset[subject]
                    label_o_o = label_o[-1]
                    risk = []
                    for ttt in range(1,signals.shape[1]):
                        risk.append(self.risk_predictor(signals[:, 0:ttt].view(1, signals.shape[0], ttt).to(self.device)).item())
                    print((max(risk) - min(risk)))
                    print('Subject ID: ', subject)
                    print('Did this patient die? ', {1:'yes',0:'no'}[label_o_o.item()>0.5])

                '''
                importance = np.zeros((self.feature_size,signals.shape[1]-1))
                mean_predicted_risk = np.zeros((self.feature_size,signals.shape[1]-1))
                std_predicted_risk = np.zeros((self.feature_size,signals.shape[1]-1))
                importance_occ = np.zeros((self.feature_size,signals.shape[1]-1))
                std_predicted_risk_occ = np.zeros((self.feature_size,signals.shape[1]-1))
                importance_comb = np.zeros((self.feature_size,signals.shape[1]-1))
                std_predicted_risk_comb = np.zeros((self.feature_size,signals.shape[1]-1))
                '''

                importance = np.zeros((self.feature_size,nt))
                mean_predicted_risk = np.zeros((self.feature_size,nt))
                std_predicted_risk = np.zeros((self.feature_size,nt))
                importance_occ = np.zeros((self.feature_size,nt))
                mean_predicted_risk_occ = np.zeros((self.feature_size,nt))
                std_predicted_risk_occ = np.zeros((self.feature_size,nt))
                importance_comb = np.zeros((self.feature_size,nt))
                mean_predicted_risk_comb = np.zeros((self.feature_size,nt))
                std_predicted_risk_comb = np.zeros((self.feature_size,nt))
                importance_su = np.zeros((self.feature_size,nt))
                mean_predicted_risk_su = np.zeros((self.feature_size,nt))
                std_predicted_risk_su = np.zeros((self.feature_size,nt))
                importance_sens = np.zeros((self.feature_size,nt))
                mean_predicted_risk_sens = np.zeros((self.feature_size,nt))
                std_predicted_risk_sens = np.zeros((self.feature_size,nt))

                f_imp = np.zeros(self.feature_size)
                f_imp_occ = np.zeros(self.feature_size)
                f_imp_comb = np.zeros(self.feature_size)
                f_imp_su = np.zeros(self.feature_size)
                f_imp_sens = np.zeros(self.feature_size)
                max_imp_total = []
                max_imp_total_occ = []
                max_imp_total_comb = []
                max_imp_total_su = []
                max_imp_total_sens = []
                legend_elements=[]

                for i, sig_ind in enumerate(range(0,self.feature_size)):
                #for i, sig_ind in enumerate(range(0,5)):
                    print('loading feature from:', ckpt_path)
                    if self.historical:
                        self.generator.load_state_dict(torch.load(os.path.join(ckpt_path,'feature_%d_generator.pt'%(sig_ind))))
                    else:
                        self.generator.load_state_dict(torch.load(os.path.join(ckpt_path,'feature_%d_generator_nohist.pt'%(sig_ind))))

                    if self.data=='simulation':
                        label, importance[i,:], mean_predicted_risk[i,:], std_predicted_risk[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode='generator', learned_risk=self.learned_risk,gt_imp = gt_importance[subject,:],tvec=tvec)
                        _, importance_occ[i, :], mean_predicted_risk_occ[i,:], std_predicted_risk_occ[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode="augmented_feature_occlusion",learned_risk=self.learned_risk,gt_imp = gt_importance[subject,:],tvec=tvec)
                        _, importance_su[i, :], mean_predicted_risk_su[i,:], std_predicted_risk_su[i,:] = self._get_feature_importance(signals,sig_ind=sig_ind, n_samples=10, mode='suresh_et_al',learned_risk=self.learned_risk,gt_imp = gt_importance[subject,:],tvec=tvec)
                        _, importance_comb[i, :], mean_predicted_risk_comb[i,:], std_predicted_risk_comb[i,:] = self._get_feature_importance(signals,sig_ind=sig_ind, n_samples=10, mode='combined',learned_risk=self.learned_risk,gt_imp= gt_importance[subject,:],tvec=tvec)

                    else:
                        label, importance[i,:], mean_predicted_risk[i,:], std_predicted_risk[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode='generator', learned_risk=self.learned_risk,tvec=tvec)
                        _, importance_occ[i, :], mean_predicted_risk_occ[i,:], std_predicted_risk_occ[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode="augmented_feature_occlusion",learned_risk=self.learned_risk,tvec=tvec)
                        _, importance_su[i, :], mean_predicted_risk_su[i,:], std_predicted_risk_su[i,:] = self._get_feature_importance(signals,sig_ind=sig_ind, n_samples=10, mode='suresh_et_al',learned_risk=self.learned_risk,tvec=tvec)
                        _, importance_comb[i, :], mean_predicted_risk_comb[i,:], std_predicted_risk_comb[i,:] = self._get_feature_importance(signals,sig_ind=sig_ind, n_samples=10, mode='combined',learned_risk=self.learned_risk,tvec=tvec)

                        #print('importance:', importance[i,:])
                    max_imp_total.append((i,max(mean_predicted_risk[i,:])))
                    max_imp_total_occ.append((i,max(mean_predicted_risk_occ[i,:])))
                    max_imp_total_comb.append((i,max(mean_predicted_risk_comb[i,:])))
                    max_imp_total_su.append((i,max(mean_predicted_risk_su[i,:])))
                    max_imp_total_sens.append((i,max(sensitivity_analysis[sub_ind,i,:])))

                    #print(label)
                    if self.data=='ghg':
                        label_scaled = loader_obj.scaler_y.inverse_transform(np.reshape(np.array(label),[-1,1]))
                        label_scaled = np.reshape(label_scaled,[1,-1])
                    else:
                        label_scaled = label

                #This chunk will measure the fraction of times top ranked feature
                #matches the groundtruth for simulated data
                if self.simulation:
                    #print([np.argmax(importance[:,t]) for t in range(47)], gt_importance[subject,:])
                    imp_acc += np.sum(np.array([int(np.argmax(importance[:,t-1])==gt_importance[subject,t-1]) for t in tvec]))
                    imp_acc_occ += np.sum(np.array([int(np.argmax(importance_occ[:,t-1])==gt_importance[subject,t-1]) for t in tvec]))
                    imp_acc_comb += np.sum(np.array([int(np.argmax(importance_comb[:,t-1])==gt_importance[subject,t-1]) for t in tvec]))
                    imp_acc_sens += np.sum(np.array([int(np.argmax(sensitivity_analysis[sub_ind,:,t-1])==gt_importance[subject,t-1]) for t in tvec]))

                    imp_acc_su += np.sum(np.array([int(np.argmax(importance_su[:,t-1])==gt_importance[subject,t-1]) for t in tvec]))
                retain_style=False
                #t = np.arange(signals.shape[1])
                if retain_style:
                    orders = np.argsort(importance, axis=0)
                    for imps in orders[-3:,:]:
                        for time in range(len(imps)):
                            imp = importance[imps[time],time]
                            texts = self.feature_map[imps[time]]
                            ax2.text(time, imp, texts)
                        ax2.set_ylim(0,np.max(importance))
                        ax2.set_xlim(0,47)

                    orders = np.argsort(importance_occ, axis=0)
                    for imps in orders[-3:,:]:
                        for time in range(len(imps)):
                            imp = importance_occ[imps[time],time]
                            texts = self.feature_map[imps[time]]
                            ax3.text(time, imp, texts)
                        ax3.set_ylim(0,np.max(importance_occ))
                        ax3.set_xlim(0,47)

                    orders = np.argsort(importance_comb, axis=0)
                    for imps in orders[-3:,:]:
                        for time in range(len(imps)):
                            imp = importance_comb[imps[time],time]
                            texts = self.feature_map[imps[time]]
                            ax4.text(time, imp, texts)
                        ax4.set_ylim(0,np.max(importance_comb))
                        ax4.set_xlim(0,47)

                else:
                    print('get feat imp')
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
                    max_imp_total_occ.sort(key=lambda pair: pair[1], reverse=True)
                    max_imp_total_comb.sort(key=lambda pair: pair[1], reverse=True)
                    max_imp_total_su.sort(key=lambda pair: pair[1], reverse=True)
                    max_imp_total_sens.sort(key=lambda pair: pair[1], reverse=True)

                    n_feats_to_plot = min(self.feature_size,3)
                    print('plotting....')
                    if self.data=='simulation':
                        ms=9;lw=3;mec='k'
                    else:
                        ms=4;lw=3;mec='k'

                    top_3_vec=[]
                    for ind,sig in max_imp_total[0:n_feats_to_plot]:
                        if self.data=='ghg':
                            if ind==0:
                                signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                            elif ind==signals.shape[0]:
                                signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:]),0)
                            else:
                                signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                            risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                            risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                            top_3_vec.append(risk-risk_removed)
                            #print(mse_vec)
   
                        ax1.plot(range(signal_scaled.shape[2]),np.array(signal_scaled[sub_ind,ind,:]), label='%s'%(self.feature_map[ind]),c=simulation_color_map[ind],marker=marker_styles_map[ind],ls=line_styles_map[ind],linewidth=lw,markersize=ms,markeredgecolor=mec)
                        ax2.errorbar(tvec, importance[ind,:], yerr=std_predicted_risk[ind,:], marker=marker_styles_map[ind],c=simulation_color_map[ind],ls=line_styles_map[ind],linewidth=lw,markersize=ms,markeredgecolor=mec)
                    
                    mse_vec.append(top_3_vec)

                    if self.data=='simulation':
                        for ind, sig in max_imp_total_su[0:n_feats_to_plot]:
                            ax3.errorbar(tvec, importance_su[ind, :], yerr=std_predicted_risk_su[ind, :], marker=marker_styles_map[ind],c=simulation_color_map[ind],ls=line_styles_map[ind],linewidth=lw,markersize=ms,markeredgecolor=mec)
                        for ind, sig in max_imp_total_sens[0:n_feats_to_plot]:
                            ax4.errorbar(tvec, importance_sens[ind, :], yerr=std_predicted_risk_sens[ind, :], marker=marker_styles_map[ind],c=simulation_color_map[ind],ls=line_styles_map[ind],linewidth=lw,markersize=ms,markeredgecolor=mec)

                    else:
                        top_3_vec_occ=[]
                        for ind, sig in max_imp_total_occ[0:n_feats_to_plot]:
                            if self.data=='ghg':
                                if ind==0:
                                    signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                                elif ind==signals.shape[0]:
                                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0].view(1,signals.shape[1]),:]),0)
                                else:
                                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                                risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                                risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                                top_3_vec_occ.append(risk-risk_removed)

                            ax3.errorbar(tvec, importance_occ[ind, :], yerr=std_predicted_risk_occ[ind, :], marker=marker_styles_map[ind],c=simulation_color_map[ind],ls=line_styles_map[ind],linewidth=lw,markersize=ms,markeredgecolor=mec)
                        mse_vec_occ.append(top_3_vec_occ)

                        top_3_vec_su=[]
                        for ind, sig in max_imp_total_su[0:n_feats_to_plot]:
                            if self.data=='ghg':
                                if ind==0:
                                    signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                                elif ind==signals.shape[0]:
                                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1])),0)
                                else:
                                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                                risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                                risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                                top_3_vec_su.append(risk-risk_removed)
                            ax4.errorbar(tvec, importance_su[ind, :], yerr=std_predicted_risk_su[ind, :], marker=marker_styles_map[ind],c=simulation_color_map[ind],ls=line_styles_map[ind],linewidth=lw,markersize=ms,markeredgecolor=mec)

                        mse_vec_su.append(top_3_vec_su)

                        top_3_vec_sens=[]
                        for ind,sig in max_imp_total_sens[0:n_feats_to_plot]:
                            if self.data=='ghg':
                                if ind==0:
                                    signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                                elif ind==signals.shape[0]:
                                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1])),0)
                                else:
                                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)

                                #signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total_sens[-1][0],:],signals[ind+1:,:]),0)
                                risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                                risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                                top_3_vec_sens.append(risk-risk_removed)

                            ax5.plot(tvec,abs(sensitivity_analysis[sub_ind,ind,:]),label='%s'%(self.feature_map[ind]),c=simulation_color_map[ind],ls=line_styles_map[ind],linewidth=lw,markersize=ms,markeredgecolor=mec)
                        mse_vec_sens.append(top_3_vec_sens)

                if not self.data=='simulation' or self.data=='simulation':
                    ax1.plot(tvec, np.reshape(label_scaled,[-1]), '--', label='Risk score',linewidth=lw,markersize=ms, c=simulation_color_map[-7])

                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width, box.height])
                #for t in range(48):
                #    ax1.axvspan(t,t+1,alpha=0.3,color=plt.get_cmap('Accent')(gt_importance[subject,t]/3),label=self.feature_map[gt_importance[subject,t]])
                
                #legend_elements = [Patch(facecolor=plt.get_cmap('Accent')(0),label=self.feature_map[0]),
                #        Patch(facecolor=plt.get_cmap('Accent')(1/3),label=self.feature_map[1]),
                #        Patch(facecolor=plt.get_cmap('Accent')(2/3),label=self.feature_map[2])]

                if self.data=='simulation' and 0:
                    ax1.axvspan(0,int(len(tvec)/3),alpha=0.3,color=simulation_color_map[gt_importance[subject,0]],label=self.feature_map[gt_importance[subject,0]] + ' on')
                    ax1.axvspan(int(len(tvec)/3),2*int(len(tvec)/3),alpha=0.3,color=simulation_color_map[gt_importance[subject,17]],label=self.feature_map[gt_importance[subject,17]] + ' on')
                    ax1.axvspan(int(2*len(tvec)/3),len(tvec),alpha=0.3,color=simulation_color_map[gt_importance[subject,33]],label=self.feature_map[gt_importance[subject,33]] + ' on')
                    #ax1.add_artist(plt.legend(handles=legend_elements,loc='center left',bbox_to_anchor=(1.05, 0.5),ncol=1, fancybox=True))
                    handles,labels=ax1.get_legend_handles_labels()
                    idx_order = [np.where([labels[xx]==ll for xx in range(len(labels))])[0][0] for ll in ['var 0', 'var 0 on','var 1', 'var 1 on' , 'var 2', 'var 2 on','Risk score']]
                    handles = [handles[idx] for idx in idx_order]
                    labels = [labels[idx] for idx in idx_order]
 
                if self.data=='simulation':
                    if int(label_o.item())==1: #true
                        ax1.axvspan(0,nt,alpha=0.3,color=simulation_color_map[-1])

                    handles,labels=ax1.get_legend_handles_labels()

                    idx_order = [np.where([labels[xx]==ll for xx in range(len(labels))])[0][0] for ll in ['var 0', 'var 1', 'var 2','Risk score']]
                    handles = [handles[idx] for idx in idx_order]
                    labels = [labels[idx] for idx in idx_order]
 
                
               #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                #ax1.legend(handles,labels,loc='lower left', bbox_to_anchor=(0.0, 1.13),ncol=7, fancybox=True,handlelength=5)               
                if 0:
                    box = ax2.get_position()
                    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),ncol=1, fancybox=True)
                    box = ax3.get_position()
                    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax3.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),ncol=1, fancybox=True)
                    box = ax4.get_position()
                    ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax4.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),ncol=1, fancybox=True)
                    box = ax5.get_position()
                    ax5.set_position([box.x0, box.y0, box.width * 0.8, box.height]) 
                    ax5.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),ncol=1, fancybox=True)
 
                fs=24
                ax1.grid(True)
                ax2.grid(True)
                ax3.grid(True)
                ax4.grid(True)
                ax4.set_xlabel('time',fontweight='bold',fontsize=22)
                ax1.set_ylabel('signal value', fontweight='bold',fontsize=22)
                ax2.set_ylabel('Imp', fontweight='bold',fontsize=22)
                ax3.set_ylabel('Imp', fontweight='bold',fontsize=22)
                ax4.set_ylabel('Imp', fontweight='bold',fontsize=22)
                ax1.tick_params(axis='both',labelsize='16')
                ax2.tick_params(axis='both',labelsize='16')
                ax3.tick_params(axis='both',labelsize='16')
                ax4.tick_params(axis='both',labelsize='16')

                #ax2.set_ylim([0,1])
                #ax3.set_ylim([0,1])
                #ax4.set_ylim([0,1])
                
                ax1.set_title('Time series signals', fontweight='bold',loc='right',fontsize=fs)
                ax2.set_title('FFC', fontweight='bold',loc='right',fontsize=fs)
 
                if self.data=='simulation':
                    #print('foo')
                    ax3.set_title('Suresh et. al.', fontweight='bold',loc='right',fontsize=fs)
                    ax4.set_title('Sensitivity Analysis', fontweight='bold',loc='right',fontsize=fs)
                else:
                    ax5.grid(True)
                    ax3.set_title('Signal importance using Augmented Feature Occlusion', fontweight='bold',loc='right')
                    ax4.set_title('Signal importance using Suresh et. al.', fontweight='bold',loc='right')
                    ax5.set_title('Signal importance using Sensitivity Analysis', fontweight='bold',loc='right')
               
                plt.savefig('/scratch/gobi1/shalmali/'+self.data+'/feature_imp_simulated_learned%d.pdf'%(subject),dpi=300,orientation='landscape',bbox_inches='tight')
                if self.data=='ghg':
                    handles,labels=ax1.get_legend_handles_labels()
                figLegend = plt.figure(figsize = (13,1.2))
                plt.figlegend(handles,labels, loc = 'upper left',ncol=4,fancybox=True, handlelength=6,fontsize='xx-large')
                figLegend.savefig('/scratch/gobi1/shalmali/'+self.data+'/legend_learned.pdf')
 
              
                if self.data=='ghg' and 0: #plot on maps
                    #fig, ax_list = plt.subplots(nrows=3,ncols=len(tvec))
                    fig, ax_list = plt.subplots(1,len(tvec),figsize=(8,2))
                    map_img = mpimg.imread('TSX/ghc_figure.png')
                    max_to_plot=2
                    colorvec = plt.get_cmap("Oranges")(np.linspace(1,2/3,max_to_plot))
                    k_count=0
                    for ind,sig in max_imp_total[0:max_to_plot]:
                        for t_num, t_val in enumerate(tvec):
                            ax_list[t_num].imshow(map_img, extent=[0,width,0,height])
                            ax_list[t_num].scatter(x=[scatter_map_ghg[str(ind+1)][0]], y=[scatter_map_ghg[str(ind+1)][1]],c=["w"],s=18,marker='s',edgecolor="r",linewidth=2)
                            if k_count==0:
                                ax_list[t_num].set_title("day %d"%(int(t_val/4)))
                        k_count+=1

                    plt.tight_layout()
                    plt.savefig('/scratch/gobi1/shalmali/ghg/feature_imp_ghg_%d.pdf'%(subject),dpi=300,orientation='landscape')

                    fig, ax_list = plt.subplots(1,len(tvec),figsize=(8,2))
                    k_count=0
                    for ind,sig in max_imp_total_occ[0:max_to_plot]:
                        for t_num, t_val in enumerate(tvec):
                            ax_list[t_num].imshow(map_img, extent=[0,width,0,height])
                            ax_list[t_num].scatter(x=[scatter_map_ghg[str(ind+1)][0]], y=[scatter_map_ghg[str(ind+1)][1]],c=["w"],s=18,marker='s',edgecolor="r",linewidth=2)
                            if k_count==0:
                                ax_list[t_num].set_title("day %d"%(t_val/4))
                        k_count+=1

                    plt.tight_layout()
                    plt.savefig('/scratch/gobi1/shalmali/ghg/feature_occ_ghg_%d.pdf'%(subject),dpi=300,orientation='landscape')
                    
                    fig, ax_list = plt.subplots(1,len(tvec),figsize=(8,2))
                    k_count=0
                    for ind,sig in max_imp_total_comb[0:max_to_plot]:
                        for t_num, t_val in enumerate(tvec):
                            ax_list[t_num].imshow(map_img, extent=[0,width,0,height])
                            ax_list[t_num].scatter(x=[scatter_map_ghg[str(ind+1)][0]], y=[scatter_map_ghg[str(ind+1)][1]],c=["w"],s=18,marker='s',edgecolor="r",linewidth=2)
                            if k_count==0:
                                ax_list[t_num].set_title("day %d"%(t_val/4))
 
                        k_count+=1

                    plt.tight_layout()
                    plt.savefig('/scratch/gobi1/shalmali/ghg/feature_imp_comb_ghg_%d.pdf'%(subject),dpi=300,orientation='landscape')

                #plt.show()
                with open('mse.pkl','wb') as f:
                    pkl.dump({'FFC': mse_vec, 'FO': mse_vec_occ, 'Su': mse_vec_su, 'Sens': mse_vec_sens},f)
            
                if self.data=='ghg':
                    print(np.shape(np.array(mse_vec)))
                    print('FFC: 1:', np.mean(abs(np.array(mse_vec)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec)[:,2])))
                    print('AFO: 1:', np.mean(abs(np.array(mse_vec_occ)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec_occ)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec_occ)[:,2])))
                    print('Su: 1:', np.mean(abs(np.array(mse_vec_su)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec_su)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec_su)[:,2])))
                    print('Sens: 1:', np.mean(abs(np.array(mse_vec_sens)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec_sens)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec_sens)[:,2])))

            print("Importance Accuracy Orig   : ", imp_acc/(len(tvec)*len(samples_to_analyse)))
            print("Importance Accuracy FeatOcc: ", imp_acc_occ/(len(tvec)*len(samples_to_analyse)))
            print("Importance Accuracy Comb   : ", imp_acc_comb/(len(tvec)*len(samples_to_analyse)))
            print("Importance Accuracy Sens   : ", imp_acc_sens/(len(tvec)*len(samples_to_analyse)))

    def train(self, n_epochs, n_features):
        for feature_to_predict in range(n_features):
            print('**** training to sample feature: ', feature_to_predict)
            if self.data=='mimic' or self.data=='ghg':
                self.generator = FeatureGenerator(self.feature_size, self.historical, data=self.data, hidden_size=50).to(self.device)
            else:
                self.generator = FeatureGenerator(self.feature_size, self.historical, data=self.data, hidden_size=10).to(self.device)
            train_feature_generator(self.generator, self.train_loader, self.valid_loader, feature_to_predict, n_epochs, self.historical, ckpt_path=os.path.join('./ckpt',self.data), data=self.data)

    def _get_feature_importance(self, signal, sig_ind, n_samples=10, mode="feature_occlusion",learned_risk=True,gt_imp=None,tvec=None):
        self.generator.eval()
        feature_dist = np.sort(np.array(self.feature_dist[:,sig_ind,:]).reshape(-1))

        risks = []
        importance = []
        mean_predicted_risk = []
        std_predicted_risk = []
        if tvec==None:
            tvec = range(1,signal.shape[1])
        #for t in range(1,signal.shape[1]):
        for t in tvec:
            if self.simulation:
                if not learned_risk:
                    risk = self.risk_predictor(signal.cpu().detach().numpy(),gt_imp,t)
                else:
                    risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t + 1)).item()
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
                if mode=="augmented_feature_occlusion":
                    prediction = torch.Tensor(np.random.choice(feature_dist).reshape(-1,)).to(self.device)
                elif mode=='suresh_et_al':
                    prediction = torch.Tensor(np.random.normal(size=[1]).reshape(-1,)).to(self.device)
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
                        predicted_risk = self.risk_predictor(predicted_signal.cpu().detach().numpy(), gt_imp, t)
                    else:
                        predicted_risk = self.risk_predictor(predicted_signal[:,0:t+1].view(1,predicted_signal.shape[0],t+1).to(self.device)).item()
                else:
                    predicted_risk = self.risk_predictor(predicted_signal[:, 0:t + 1].view(1, predicted_signal.shape[0], t + 1).to(self.device)).item()
                    # predicted_risk = self.risk_predictor(predicted_signal.view(1, predicted_signal.shape[0], predicted_signal.shape[1]).to(self.device)).item()
                #print('predicted risk',predicted_risk)
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
            self.generator.load_state_dict(torch.load(os.path.join(ckpt_path,'feature_%d_generator.pt'%(sig))))
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

