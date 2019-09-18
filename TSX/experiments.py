import torch
import os,sys,glob
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, train_model_rt, train_model_rt_rg, test, test_model_rt_rg, logistic
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE
from TSX.generator import FeatureGenerator, train_joint_feature_generator, train_feature_generator, CarryForwardGenerator, DLMGenerator, JointFeatureGenerator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.image as mpimg
import json

import lime
import lime.lime_tabular

#generic plot configs
line_styles_map=['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--','-.',':']
marker_styles_map=['o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h']

#mimic plot configs
xkcd_colors = mcolors.XKCD_COLORS
color_map = [list(xkcd_colors.keys())[k] for k in
             np.random.choice(range(len(xkcd_colors)), 28, replace=False)]
color_map = ['#990000', '#C20088', '#0075DC', '#993F00', '#4C005C', '#191919', '#FF0010', '#2BCE48', '#FFCC99', '#808080',
             '#740AFF', '#8F7C00', '#9DCC00', '#F0A3FF', '#94FFB5', '#FFA405', '#FFA8BB', '#426600', '#005C31', '#5EF1F2',
             '#00998F', '#E0FF66', '#003380', '#990000', '#FFFF80', '#FF5005', '#FFFF00','#FF0010', '#FFCC99']

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']

feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

#simulation plot configs
feature_map_simulation = ['feature 0', 'feature 1', 'feature 2']

simulation_color_map = ['#e6194B', '#469990', '#000000','#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',  '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#3cb44b','#ffe119']

MIMIC_TEST_SAMPLES = [4387, 481]

#ghg plot configs
feature_map_ghg = [str(i+1) for i in range(15)]
scatter_map_ghg={}
for i in range(15):
    scatter_map_ghg[feature_map_ghg[i]]=[]
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
    def __init__(self, train_loader, valid_loader, test_loader, data='mimic'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.data = data

    @abstractmethod
    def run(self):
        raise RuntimeError('Function not implemented')

    def train(self, n_epochs, learn_rt=False):
        if self.data=='mimic':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-3)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        
        if not learn_rt:
            train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment, data=self.data)
            # Evaluate performance on held-out test set
            _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            print('\nFinal performance on held out test set ===> AUC: ', auc_test)
        else:
            if self.data == 'mimic' or self.data=='simulation':
                train_model_rt(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment,data=self.data)
            elif self.data == 'ghg':
                train_model_rt_rg(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment,data=self.data)


class Baseline(Experiment): # TODO: Add checkpoint point and experiment name as attributes of the base class
    """ Baseline mortality prediction using a logistic regressions model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, experiment='baseline',data='mimic'):
        super(Baseline, self).__init__(train_loader, valid_loader, test_loader)
        self.model = LR(feature_size).to(self.device)
        self.experiment = experiment
        self.data=data

    def run(self, train):
        if train:
            self.train(n_epochs=250)
        else:
            if os.path.exists('./ckpt/' + self.data + '/' +  str(self.experiment) + '.pt'):
                self.model.load_state_dict(torch.load('./ckpt/'+ self.data + '/' + str(self.experiment) + '.pt'))
                _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                print('Loading model with AUC: ', auc_test)
            else:
                raise RuntimeError('No saved checkpoint for this model')


class EncoderPredictor(Experiment):
    """ Baseline mortality prediction using an encoder to encode patient status, and a risk predictor to predict risk of mortality
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor',simulation=False,data='mimic'):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader, data=data)
        self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=False,data=data)
        self.experiment = experiment
        self.simulation = simulation
        self.data = data
        self.ckpt_path='./ckpt/' + self.data

    def run(self, train,n_epochs=60):
        if train:
            self.train(n_epochs=n_epochs, learn_rt=self.data!='mimc')
        else:
            if os.path.exists('./ckpt/' + self.data + '/' + str(self.experiment) + '.pt'):
                self.model.load_state_dict(torch.load(os.path.join('./ckpt/' + self.data, str(self.experiment) + '.pt')))

                if not self.data=='ghg':
                    _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                else:#ghg is regression
                    test_loss = test_rt(self.test_loader, self.model, self.device)
            else:
                raise RuntimeError('No saved checkpoint for this model')

    def train(self, n_epochs, learn_rt=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-3)

        if not learn_rt:
            train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                            self.experiment, data=self.data)
            # Evaluate performance on held-out test set
            _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            print('\nFinal performance on held out test set ===> AUC: ', auc_test)
        else:
            if self.data=='simulation':
                print('training rt')
                train_model_rt(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                            self.experiment, data=self.data)
            else:
                #only for ghg data
                train_model_rt_rg(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                               self.experiment,data=self.data)


class BaselineExplainer(Experiment):
    """ Baseline explainability methods
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, data_class, experiment='baseline_explainer', data='mimic', baseline_method='lime'):
        super(BaselineExplainer, self).__init__(train_loader, valid_loader, test_loader, data=data)
        self.experiment = experiment
        self.data_class = data_class
        self.ckpt_path = os.path.join('./ckpt/', self.data)
        self.baseline_method = baseline_method
        self.input_size = feature_size
        if data == 'mimic':
            self.timeseries_feature_size = feature_size - 4
        else:
            self.timeseries_feature_size = feature_size

        # Build the risk predictor and load checkpoint
        with open('config.json') as config_file:
            configs = json.load(config_file)[data]['risk_predictor']
        if self.data == 'simulation':
            if not self.learned_risk:
                self.risk_predictor = lambda signal,t:logistic(2.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t] - 1))
            else:
                self.risk_predictor = EncoderRNN(configs['feature_size'], hidden_size=configs['encoding_size'],
                                                 rnn=configs['rnn_type'], regres=True, return_all=False, data=data)
            self.feature_map = feature_map_simulation
        else:
            if self.data == 'mimic':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
                                                 rnn=configs['rnn_type'], regres=True, data=data)
                self.feature_map = feature_map_mimic
                self.risk_predictor = self.risk_predictor.to(self.device)
            elif self.data == 'ghg':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
                                                 rnn=configs['rnn_type'], regres=True, data=data)
                self.feature_map = feature_map_ghg
                self.risk_predictor = self.risk_predictor.to(self.device)
        self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor.pt')))
        self.risk_predictor.to(self.device)
        self.risk_predictor.eval()

    def predictor_wrapper(self, sample):
        """
        In order to use the lime explainer library we need to go back and forth between numpy library (compatible with Lime)
        and torch (Compatible with the predictor model). This wrapper helps with this
        :param sample: input sample for the predictor (type: numpy array)
        :return: one-hot model output (type: numpy array)
        """
        torch_in = torch.Tensor(sample).reshape(len(sample),-1,1)
        torch_in.to(self.device)
        out = self.risk_predictor(torch_in)
        one_hot_out = np.concatenate((out.detach().cpu().numpy(), out.detach().cpu().numpy()), axis=1)
        one_hot_out[:,1] = 1-one_hot_out[:,0]
        return one_hot_out

    def run(self, train, n_epochs=60):
        self.train(n_epochs=n_epochs, learn_rt=self.data=='ghg')
        testset = list(self.test_loader.dataset)
        test_signals = torch.stack(([x[0] for x in testset])).to(self.device)
        matrix_test_dataset = test_signals.mean(dim=2).cpu().numpy()
        for test_sample in MIMIC_TEST_SAMPLES:
            exp = self.explainer.explain_instance(matrix_test_dataset[test_sample], self.predictor_wrapper, num_features=4, top_labels=2)
            print("Most important features for sample %d: "%(test_sample), exp.as_list())

    def train(self, n_epochs, learn_rt=False):
        trainset = list(self.train_loader.dataset)
        signals = torch.stack(([x[0] for x in trainset])).to(self.device)
        matrix_train_dataset = signals.mean(dim=2).cpu().numpy()
        if self.baseline_method == 'lime':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(matrix_train_dataset, feature_names=self.feature_map+['gender', 'age', 'ethnicity', 'first_icu_stay'], discretize_continuous=True)


class FeatureGeneratorExplainer(Experiment):
    """ Experiment for generating feature importance over time using a generative model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, patient_data, generator_hidden_size=80, prediction_size=1, historical=False, generator_type='RNN_generator', experiment='feature_generator_explainer', data='mimic', conditional=True):
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
        super(FeatureGeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader, data)
        self.generator_type = generator_type
        if self.generator_type == 'RNN_generator':
            self.generator = FeatureGenerator(feature_size, historical, hidden_size=generator_hidden_size, prediction_size=prediction_size,data=data,conditional=conditional).to(self.device)
            self.conditional=conditional
        elif self.generator_type == 'carry_forward_generator':
            self.generator = CarryForwardGenerator(feature_size).to(self.device)
        elif self.generator_type == 'joint_RNN_generator':
            self.generator = JointFeatureGenerator(feature_size, data=data).to(self.device) # TODO setup the right encoding size
        elif self.generator_type == 'dlm_joint_generator':
            self.generator = DLMGenerator(feature_size).to(self.device) # TODO setup the right encoding size
        else:
            raise RuntimeError('Undefined generator!')

        if data == 'mimic':
            self.timeseries_feature_size = feature_size - 4
        else:
            self.timeseries_feature_size = feature_size

        self.feature_size = feature_size
        self.input_size = feature_size
        self.patient_data = patient_data
        self.experiment = experiment
        self.historical = historical
        self.simulation = self.data=='simulation'
        self.spike_data=0
        self.prediction_size = prediction_size
        self.generator_hidden_size = generator_hidden_size
        if self.generator_type!='RNN_generator':
            self.conditional=None

        #this is used to see the difference between true risk vs learned risk for simulations
        self.learned_risk = True
        trainset = list(self.train_loader.dataset)
        self.feature_dist = torch.stack([x[0] for x in trainset])
        if self.data == 'mimic':
            self.feature_dist_0 = torch.stack([x[0] for x in trainset if x[1]==0])
            self.feature_dist_1 = torch.stack([x[0] for x in trainset if x[1]==1])
        else:
            self.feature_dist_0=self.feature_dist
            self.feature_dist_1=self.feature_dist

        # TODO: instead of hard coding read from json
        if self.data=='simulation':
            if not self.learned_risk:
                self.risk_predictor = lambda signal,t:logistic(2.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t] - 1))
            else:
                self.risk_predictor = EncoderRNN(feature_size,hidden_size=100,rnn='GRU',regres=True, return_all=False,data=data)
            self.feature_map = feature_map_simulation
        else:
            if self.data=='mimic':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=150, rnn='GRU', regres=True,data=data)
                self.feature_map = feature_map_mimic
                self.risk_predictor = self.risk_predictor.to(self.device)
            elif self.data=='ghg':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=100, rnn='GRU', regres=True,data=data)
                self.feature_map = feature_map_ghg
                self.risk_predictor = self.risk_predictor.to(self.device)

    def run(self, train,n_epochs=80):
        """ Run feature generator experiment
        :param train: (boolean) If True, train the generators, if False, use saved checkpoints
        """
        if train and self.generator_type!='carry_forward_generator':
           self.train(n_features=self.timeseries_feature_size, n_epochs=n_epochs)
        else:
            ckpt_path = os.path.join('./ckpt',self.data)
            if self.historical:
                check_path = glob.glob(os.path.join(ckpt_path,'*_generator.pt'))[0]
            else:
                check_path = glob.glob(os.path.join(ckpt_path,'*_generator.pt'))[0]

            if not os.path.exists(check_path):
                raise RuntimeError('No saved checkpoint for this model')
            else:
                if not self.data=='simulation':
                    self.risk_predictor.load_state_dict(torch.load(os.path.join(ckpt_path,'risk_predictor.pt')))
                    self.risk_predictor = self.risk_predictor.to(self.device)
                    self.risk_predictor.eval()
                else: #simulated data
                    if self.learned_risk:
                        self.risk_predictor.load_state_dict(torch.load(os.path.join(ckpt_path,'risk_predictor.pt')))
                        self.risk_predictor = self.risk_predictor.to(self.device)
                        self.risk_predictor.eval()
                #print('Generator test loss: ', gen_test_loss)

            testset = list(self.test_loader.dataset)
            if self.data=='simulation':
                if self.spike_data==1:
                    with open(os.path.join('./data_generator/data/simulated_data/thresholds_test.pkl'), 'rb') as f:
                        th = pkl.load(f)

                    with open(os.path.join('./data_generator/data/simulated_data/gt_test.pkl'), 'rb') as f:
                        gt_importance = pkl.load(f)#Type dmesg and check the last few lines of output. If the disc or the connection to it is failing, it'll be noted there.load(f)
                else:
                    with open(os.path.join('./data/simulated_data/state_dataset_states_test.pkl'),'rb') as f:
                        gt_importance = pkl.load(f)

                #For simulated data this is the last entry - end of 48 hours that's the actual outcome
                label = np.array([x[1][-1] for x in testset])
                #print(label)
                high_risk = np.where(label==1)[0]
                #samples_to_analyse = np.random.choice(high_risk, 10, replace=False)
                samples_to_analyse = [101, 48, 88, 192, 143, 166, 18, 58, 172, 132]
            else:
                if self.data=='mimic':
                    samples_to_analyse = MIMIC_TEST_SAMPLES
                elif self.data=='ghg':
                    label = np.array([x[1][-1] for x in testset])
                    high_risk = np.arange(label.shape[0])
                    samples_to_analyse = np.random.choice(high_risk, len(high_risk), replace=False)

            ## Sensitivity analysis as a baseline
            signal = torch.stack([testset[sample][0] for sample in samples_to_analyse])

            #Some setting up for ghg data
            if self.data=='ghg':
                label_tch = torch.stack([testset[sample][1] for sample in samples_to_analyse])
                signal_scaled = self.patient_data.scaler_x.inverse_transform(np.reshape(signal.cpu().detach().numpy(),[len(samples_to_analyse),-1]))
                signal_scaled = np.reshape(signal_scaled,signal.shape)
                label_scaled = self.patient_data.scaler_y.inverse_transform(np.reshape(label_tch.cpu().detach().numpy(),[len(samples_to_analyse),-1]))
                label_scaled = np.reshape(label_scaled,label_tch.shape)
                #label_scaled = label_tch.cpu().detach().numpy()

                tvec = [int(x) for x in np.linspace(1,signal.shape[2]-1,5)]
            else:
                tvec = list(range(1,signal.shape[2]+1))
                signal_scaled = signal

            nt = len(tvec)
            sensitivity_analysis = np.zeros((signal.shape))

            if not self.data=='simulation':
                if self.data=='mimic' or self.data=='ghg':
                    self.risk_predictor.train()
                    for t_ind,t in enumerate(tvec):
                        signal_t = torch.Tensor(signal[:,:,:t]).to(self.device).requires_grad_()
                        out = self.risk_predictor(signal_t)
                        for s in range(len(samples_to_analyse)):
                            out[s, 0].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:,t-1].cpu().detach().numpy()
                            signal_t.grad.data.zero_()
                self.risk_predictor.eval()
            else:
                if not self.learned_risk:
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
            self.risk_predictor.load_state_dict(torch.load(os.path.join('./ckpt',self.data,'risk_predictor.pt')))
            self.risk_predictor.to(self.device)
            self.risk_predictor.eval()
            if self.data=='mimic':
                signals_to_analyze = range(0, self.timeseries_feature_size)
            elif self.data=='simulation':
                signals_to_analyze = range(0,3)
            elif self.data=='ghg':
                signals_to_analyze = range(0,15)

            if self.data=='ghg':
                #GHG experiment variables
                mse_vec=[]
                mse_vec_occ=[]
                mse_vec_su=[]
                mse_vec_comb=[]
                mse_vec_sens=[]

                # Replace and Predict Experiment
                self.replace_and_predict(signals_to_analyze, sensitivity_analysis, data=self.data, tvec=tvec)
            else:
                for sub_ind, subject in enumerate(samples_to_analyse):
                    self.plot_baseline(subject, signals_to_analyze, sensitivity_analysis[sub_ind,:,:],data=self.data,gt_importance_subj=gt_importance[subject,:])

    def plot_baseline(self, subject, signals_to_analyze, sensitivity_analysis_importance, retain_style=False, n_important_features=3,data='mimic',gt_importance_subj=None):
        """ Plot importance score across all baseline methods
        :param subject: ID of the subject to analyze
        :param signals_to_analyze: list of signals to include in importance analysis
        :param sensitivity_analysis_importance: Importance score over time under sensitivity analysis for the subject
        :param retain_style: Plotting mode. If true, top few important signal names will be plotted at every time point. Only true for MIMIC
        :param n_important_features: Number of important signals to plot
        """
        if not os.path.exists('./examples'):
            os.mkdir('./examples')
        if not os.path.exists(os.path.join('./examples',data)):
            os.mkdir(os.path.join('./examples',data))

        testset = list(self.test_loader.dataset)
        signals, label_o = testset[subject]
        print('Subject ID: ', subject)
        if data=='mimic':
            print('Did this patient die? ', {1: 'yes', 0: 'no'}[label_o.item()])

        importance = np.zeros((self.timeseries_feature_size, signals.shape[1]-1))
        mean_predicted_risk = np.zeros((self.timeseries_feature_size, signals.shape[1]-1))
        std_predicted_risk = np.zeros((self.timeseries_feature_size, signals.shape[1]-1))
        importance_occ = np.zeros((self.timeseries_feature_size, signals.shape[1]-1))
        std_predicted_risk_occ = np.zeros((self.timeseries_feature_size, signals.shape[1]-1))
        importance_occ_aug = np.zeros((self.timeseries_feature_size, signals.shape[1]-1))
        std_predicted_risk_occ_aug = np.zeros((self.timeseries_feature_size, signals.shape[1]-1))
        max_imp_FCC = []
        max_imp_occ = []
        max_imp_occ_aug = []
        max_imp_sen = []
        # TODO for joint models avoid iteratig over all samples
        for i, sig_ind in enumerate(signals_to_analyze):
            #state = np.zeros((signals.shape[1]-1))
            #print(gt_importance.shape)
            #state[gt_importance[sig_ind,1:,1]==1] = 1

            if not self.generator_type=='carry_forward_generator':
                if 'joint' in self.generator_type:
                    self.generator.load_state_dict(
                        torch.load(os.path.join('./ckpt/%s/%s.pt'%(self.data, self.generator_type))))
                else:
                    if self.historical:
                        if data=='mimic':
                            self.generator.load_state_dict(
                            torch.load(os.path.join('./ckpt',data,'%s_%s.pt' % (feature_map_mimic[sig_ind], self.generator_type))))
                        elif data=='simulation':
                            self.generator.load_state_dict(
                            torch.load(os.path.join('./ckpt',data,'%s_%s.pt' % (str(sig_ind), self.generator_type))))
                        elif data=='ghg':
                            self.generator.load_state_dict(
                            torch.load(os.path.join('./ckpt',data,'%s_%s.pt' % (str(sig_ind),self.generator_type))))
                    else:
                        if data=='mimic':
                            self.generator.load_state_dict(
                            torch.load(os.path.join('.ckpt',data,'%s_%s_nohist.pt' % (feature_map_mimic[sig_ind], self.generator_type))))
                        elif data=='simulation':
                            self.generator.load_state_dict(
                            torch.load(os.path.join('./ckpt',data,'%s_generator_nohist.pt' % (str(sig_ind)))))
                        elif data=='ghg':
                            self.generator.load_state_dict(
                            torch.load(os.path.join('./ckpt',data,'%s_generator_nohist.pt' % (str(sig_ind)))))

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

        with open(os.path.join('./examples',data,'results_'+str(subject)+'.pkl'),'wb') as f:
            pkl.dump({'FFC': {'imp':importance,'std':std_predicted_risk}, 'Suresh_et_al':{'imp':importance_occ,'std':std_predicted_risk_occ}, 'AFO': {'imp':importance_occ_aug,'std': std_predicted_risk_occ_aug}, 'Sens': {'imp': sensitivity_analysis_importance,'std':[]}, 'gt': gt_importance_subj},f,protocol=pkl.HIGHEST_PROTOCOL)

        #return

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        t = np.arange(signals.shape[1]-1)
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
        l_style = ['solid']#'-.', '--', ':']
        important_signals = []

        # TODO Remove first important assignments for FFC, becuase poor quality of the generator results in different scaling
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
        for ttt in t:
            if gt_importance_subj[ttt+1]==1:
                ax1.axvspan(ttt,ttt+1,facecolor='g',alpha=0.3)
            else:
                ax1.axvspan(ttt,ttt+1,facecolor='y',alpha=0.3)

        for ttt in t:
            if gt_importance_subj[ttt+1]==1:
                ax2.axvspan(ttt,ttt+1,facecolor='g',alpha=0.3)
            else:
                ax2.axvspan(ttt,ttt+1,facecolor='y',alpha=0.3)

        for ttt in t:
            if gt_importance_subj[ttt+1]==1:
                ax3.axvspan(ttt,ttt+1,facecolor='g',alpha=0.3)
            else:
                ax3.axvspan(ttt,ttt+1,facecolor='y',alpha=0.3)

        for ttt in t:
            if gt_importance_subj[ttt+1]==1:
                ax4.axvspan(ttt,ttt+1,facecolor='g',alpha=0.3)
            else:
                ax4.axvspan(ttt,ttt+1,facecolor='y',alpha=0.3)

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
            #ax1.plot(np.array(signals[ref_ind, 1:] / max(abs(signals[ref_ind, 1:]))), linewidth=3,
            #         linestyle=l_style[i % len(l_style)], color=c,
            #         label='%s' % (self.feature_map[ref_ind]))
            ax1.plot(np.array(signals[ref_ind, 1:]), linewidth=3,
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
        ax2.set_title('FFC', fontweight='bold', fontsize=34) # TODO change the title depending on the type of generator is being used
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
        plt.savefig(os.path.join('./examples',data,'feature_%d_%s.pdf' %(subject, self.generator_type)), dpi=300, orientation='landscape',
                    bbox_inches='tight')
        fig_legend = plt.figure(figsize=(13, 1.2))
        handles, labels = ax1.get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
        fig_legend.savefig(os.path.join('./examples', data, 'legend_%d_%s.pdf' %(subject, self.generator_type)), dpi=300, bbox_inches='tight')

    def replace_and_predict(self, signals_to_analyze, sensitivity_analysis_importance, n_important_features=3, data='ghg', tvec=None):
        mse_vec=[]
        mse_vec_occ=[]
        mse_vec_su=[]
        mse_vec_comb=[]
        mse_vec_sens=[]

        nt = len(tvec)
        testset = list(self.test_loader.dataset)
        for sub_ind, subject in enumerate(signals_to_analyze):#range(30):
            signals, label_o = testset[subject]
            label_o_o = label_o[-1]
            risk = []
            for ttt in range(1,signals.shape[1]):
                risk.append(self.risk_predictor(signals[:, 0:ttt].view(1, signals.shape[0], ttt).to(self.device)).item())

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

            ckpt_path='./ckpt/' + data + '/'
            for i, sig_ind in enumerate(range(0,self.feature_size)):
                print('loading feature from:', ckpt_path)
                if not self.generator_type=='carry_forward_generator':
                    if 'joint' in self.generator_type:
                        self.generator.load_state_dict(torch.load(os.path.join(ckpt_path, '%s.pt'%self.generator_type)))
                    else:
                        if self.historical:
                            self.generator.load_state_dict(torch.load(os.path.join(ckpt_path,'%d_%s.pt'%(sig_ind, self.generator_type))))
                        else:
                            self.generator.load_state_dict(torch.load(os.path.join(ckpt_path,'%d_%s_nohist.pt'%(sig_ind, self.generator_type))))

                label, importance[i,:], mean_predicted_risk[i,:], std_predicted_risk[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode='generator',tvec=tvec)
                _, importance_occ[i, :], mean_predicted_risk_occ[i,:], std_predicted_risk_occ[i,:] = self._get_feature_importance(signals, sig_ind=sig_ind, n_samples=10, mode="feature_occlusion",tvec=tvec)
                _, importance_su[i, :], mean_predicted_risk_su[i,:], std_predicted_risk_su[i,:] = self._get_feature_importance(signals,sig_ind=sig_ind, n_samples=10, mode='augmented_feature_occlusion',tvec=tvec)
                #_, importance_comb[i, :], mean_predicted_risk_comb[i,:], std_predicted_risk_comb[i,:] = self._get_feature_importance(signals,sig_ind=sig_ind, n_samples=10, mode='combined',learned_risk=self.learned_risk,tvec=tvec)

                    #print('importance:', importance[i,:])
                max_imp_total.append((i,max(mean_predicted_risk[i,:])))
                max_imp_total_occ.append((i,max(mean_predicted_risk_occ[i,:])))
                max_imp_total_su.append((i,max(mean_predicted_risk_su[i,:])))
                max_imp_total_sens.append((i,max(sensitivity_analysis_importance[sub_ind,i,:])))

                #print(label)
                label_scaled = self.patient_data.scaler_y.inverse_transform(np.reshape(np.array(label),[-1,1]))
                label_scaled = np.reshape(label_scaled,[1,-1])

            max_imp = np.argmax(importance,axis=0)
            for im in max_imp:
                f_imp[im] += 1
            max_imp_occ = np.argmax(importance_occ,axis=0)
            for im in max_imp_occ:
                f_imp_occ[im] += 1

            ## Pick the most influential signals and plot their importance over time
            max_imp_total.sort(key=lambda pair: pair[1], reverse=True)
            max_imp_total_occ.sort(key=lambda pair: pair[1], reverse=True)
            #max_imp_total_comb.sort(key=lambda pair: pair[1], reverse=True)
            max_imp_total_su.sort(key=lambda pair: pair[1], reverse=True)
            max_imp_total_sens.sort(key=lambda pair: pair[1], reverse=True)

            n_feats_to_plot = min(self.feature_size, n_important_features)
            ms=4;lw=3;mec='k'

            top_3_vec=[]
            for ind,sig in max_imp_total[0:n_feats_to_plot]:
                if ind==0:
                    signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                elif ind==signals.shape[0]:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:]),0)
                else:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                top_3_vec.append(risk-risk_removed)

            mse_vec.append(top_3_vec)

            top_3_vec_occ=[]
            for ind, sig in max_imp_total_occ[0:n_feats_to_plot]:
                if ind==0:
                    signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                elif ind==signals.shape[0]:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0].view(1,signals.shape[1]),:]),0)
                else:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                top_3_vec_occ.append(risk-risk_removed)

            mse_vec_occ.append(top_3_vec_occ)

            top_3_vec_su=[]
            for ind, sig in max_imp_total_su[0:n_feats_to_plot]:
                if ind==0:
                    signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                elif ind==signals.shape[0]:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1])),0)
                else:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                top_3_vec_su.append(risk-risk_removed)

            mse_vec_su.append(top_3_vec_su)

            top_3_vec_sens=[]
            for ind,sig in max_imp_total_sens[0:n_feats_to_plot]:
                if ind==0:
                    signal_removed = torch.cat((signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)
                elif ind==signals.shape[0]:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1])),0)
                else:
                    signal_removed = torch.cat((signals[:ind,:],signals[max_imp_total[-1][0],:].view(1,signals.shape[1]),signals[ind+1:,:]),0)

                risk = self.risk_predictor(signals.view(1, signals.shape[0], signals.shape[1])).item()
                risk_removed = self.risk_predictor(signal_removed.view(1, signals.shape[0], signals.shape[1])).item()
                top_3_vec_sens.append(risk-risk_removed)

            mse_vec_sens.append(top_3_vec_sens)

            #fig, ax_list = plt.subplots(nrows=3,ncols=len(tvec))
            fig, ax_list = plt.subplots(1,len(tvec),figsize=(8,2))
            map_img = mpimg.imread('./results/ghc_figure.png')
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


            with open('./results/ghg_mse.pkl','wb') as f:
                pkl.dump({'FFC': mse_vec, 'FO': mse_vec_occ, 'Su': mse_vec_su, 'Sens': mse_vec_sens},f)
        
            if self.data=='ghg':
                print(np.shape(np.array(mse_vec)))
                print('FFC: 1:', np.mean(abs(np.array(mse_vec)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec)[:,2])))
                print('FO: 1:', np.mean(abs(np.array(mse_vec_occ)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec_occ)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec_occ)[:,2])))
                print('AFO: 1:', np.mean(abs(np.array(mse_vec_su)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec_su)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec_su)[:,2])))
                print('Sens: 1:', np.mean(abs(np.array(mse_vec_sens)[:,0])), '2nd:', np.mean(abs(np.array(mse_vec_sens)[:,1])), '3rd:', np.mean(abs(np.array(mse_vec_sens)[:,2])))

    def train(self, n_epochs, n_features):
        if 'joint' in self.generator_type:
            train_joint_feature_generator(self.generator, self.train_loader, self.valid_loader, generator_type=self.generator_type , n_epochs=n_epochs)
        else:
            for feature_to_predict in range(0,n_features):
                print('**** training to sample feature: ', feature_to_predict)
                self.generator = FeatureGenerator(self.feature_size, self.historical, hidden_size=self.generator_hidden_size, prediction_size=self.prediction_size,data=self.data,conditional=self.conditional).to(self.device)
                train_feature_generator(self.generator, self.train_loader, self.valid_loader, self.generator_type, feature_to_predict, n_epochs=n_epochs, historical=self.historical)

    def _get_feature_importance(self, signal, sig_ind, n_samples=10, mode="feature_occlusion", learned_risk=True, tvec=None, cond_one=False):
        self.generator.eval()
        feature_dist = np.sort(np.array(self.feature_dist[:,sig_ind,:]).reshape(-1))
        feature_dist_0 = (np.array(self.feature_dist_0[:, sig_ind, :]).reshape(-1))
        feature_dist_1 = (np.array(self.feature_dist_1[:, sig_ind, :]).reshape(-1))

        risks = []
        importance = []
        mean_predicted_risk = []
        std_predicted_risk = []
        if tvec is None:
            tvec = range(1,signal.shape[1])
        for t in tvec:
            if self.simulation:
                if not learned_risk:
                    risk = self.risk_predictor(signal.cpu().detach().numpy(), t)
                else:
                    risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t + 1)).item()
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
                    prediction = torch.Tensor(np.array([np.random.uniform(-1,1)]).reshape(-1)).to(self.device)
                    predicted_signal = signal[:, 0:t + self.generator.prediction_size].clone()
                    predicted_signal[:, t:t + self.generator.prediction_size] = torch.cat((signal[:sig_ind,
                                                                                           t:t + self.generator.prediction_size],
                                                                                           prediction.view(1, -1),
                                                                                           signal[sig_ind + 1:,
                                                                                           t:t + self.generator.prediction_size]),
                                                                                          0)
                elif mode=="augmented_feature_occlusion":
                    if self.risk_predictor(signal[:,0:t].view(1, signal.shape[0], t)).item() > 0.5:
                        prediction = torch.Tensor(np.array(np.random.choice(feature_dist_0)).reshape(-1,)).to(self.device)
                    else:
                        prediction = torch.Tensor(np.array(np.random.choice(feature_dist_1)).reshape(-1,)).to(self.device)
                    predicted_signal = signal[:, 0:t + self.generator.prediction_size].clone()
                    predicted_signal[:, t:t + self.generator.prediction_size] = torch.cat((signal[:sig_ind,
                                                                                           t:t + self.generator.prediction_size],
                                                                                           prediction.view(1, -1),
                                                                                           signal[sig_ind + 1:,
                                                                                           t:t + self.generator.prediction_size]),
                                                                                          0)
                elif mode=="generator" or mode=="combined":
                    # TODO: This is an aweful way of conditioning on single variable!!!! Fix it
                    if cond_one:
                        predicted_signal_t, _ = self.generator(signal_known.view(1, -1),
                                                       signal[:, 0:t].view(1, signal.size(0), t), sig_ind, signal[sig_ind, t].view(1, -1), cond_one)
                        predicted_signal = signal[:,0:t+self.generator.prediction_size].clone()
                        predicted_signal[:,t:t+self.generator.prediction_size] = predicted_signal_t.view(-1,1)
                    else:
                        prediction, _ = self.generator(signal_known.view(1,-1), signal[:, 0:t].view(1,signal.size(0),t), sig_ind)
                        if mode=="combined":
                            if self.risk_predictor(signal[:,0:t].view(1, signal.shape[0], t)).item() > 0.5:
                                prediction = torch.Tensor(self._find_closest(feature_dist_0, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)
                            else:
                                prediction = torch.Tensor(self._find_closest(feature_dist_1, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)
                            # prediction = torch.Tensor(self._find_closest(feature_dist, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)

                        predicted_signal = signal[:,0:t+self.generator.prediction_size].clone()
                        predicted_signal[:,t:t+self.generator.prediction_size] = torch.cat((signal[:sig_ind,t:t+self.generator.prediction_size], prediction.view(1,-1), signal[sig_ind+1:,t:t+self.generator.prediction_size]),0)
                if self.simulation:
                    if not learned_risk:
                        predicted_risk = self.risk_predictor(predicted_signal.cpu().detach().numpy(), t)
                    else:
                        predicted_risk = self.risk_predictor(predicted_signal[:,0:t+self.generator.prediction_size].view(1,predicted_signal.shape[0],t+self.generator.prediction_size).to(self.device)).item()
                else:
                    predicted_risk = self.risk_predictor(predicted_signal[:, 0:t + self.generator.prediction_size].view(1, predicted_signal.shape[0], t + self.generator.prediction_size).to(self.device)).item()
                    # predicted_risk = self.risk_predictor(predicted_signal.view(1, predicted_signal.shape[0], predicted_signal.shape[1]).to(self.device)).item()
                predicted_risks.append(predicted_risk)
            risks.append(risk)
            predicted_risks = np.array(predicted_risks)
            #print(predicted_risks.shape)
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
            if self.historical:
                self.generator.load_state_dict(torch.load(os.path.join(self.ckpt_path, '%d_%s.pt' % (sig, self.generator_type))))
            else:
                self.generator.load_state_dict(torch.load(os.path.join(self.ckpt_path, '%d_%s_nohist.pt'%(sig, self.generator_type))))
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
            self.risk_predictor.train()
            for t in range(1,test_signals.size(2)):
                signal_t = torch.Tensor(test_signals[:,:,:t+1]).to(self.device).requires_grad_()
                out = self.risk_predictor(signal_t)
                for s in range(len(ind_list)):
                    out[s, 0].backward(retain_graph=True)
                    sensitivity_analysis[s,:,t] = signal_t.grad.data[s,:,t].cpu().detach().numpy()
                    signal_t.grad.data.zero_()
            self.risk_predictor.eval()

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
            if not os.path.exists('./interventions'):
                os.mkdir("./interventions")
            df.to_pickle("./interventions/int_%d.pkl"%(intervention_ID))

    def plot_summary_stat(self, intervention_ID=1):
        df = pd.read_pickle("./interventions/int_%d.pkl" % (intervention_ID))
        fcc_df = df.loc[df['method']=='FCC']
        occ_df = df.loc[df['method'] == 'f_occ']
        sen_df = df.loc[df['method'] == 'sensitivity']
        fcc_dist = np.sort(np.array(fcc_df[['top1','top2','top3']]).reshape(-1,))
        occ_dist = np.sort(np.array(occ_df[['top1', 'top2', 'top3']]).reshape(-1, ))
        sen_dist = np.sort(np.array(sen_df[['top1', 'top2', 'top3']]).reshape(-1, ))


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
        f.set_figwidth(15)
        if not os.path.exists('./plots/distributions'):
            os.mkdir('./plots/distributions')
        plt.savefig('./plots/distributions/top_%s'%(intervention_list[intervention_ID]), dpi=300, bbox_inches='tight')


        f, (ax1,ax2,ax3) = plt.subplots(3, sharex=True)
        ax1.bar(self.feature_map, self._find_count(fcc_dist))
        ax2.bar(self.feature_map, self._find_count(occ_dist))
        ax3.bar(self.feature_map, self._find_count(sen_dist))
        ax1.set_title('FFC importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        ax2.set_title('feature occlusion importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        ax3.set_title('sensitivity analysis importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        f.set_figheight(10)
        f.set_figwidth(20)
        plt.savefig('./plots/distributions/%s'%(intervention_list[intervention_ID]))

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
        if target <= arr[0]:
            return arr[0]
        if target >= arr[n - 1]:
            return arr[n - 1]
        i = 0
        j = n
        mid = 0
        while i < j:
            mid = (i + j) // 2
            if arr[mid] == target:
                return arr[mid]
            if target < arr[mid]:
                if mid > 0 and target > arr[mid - 1]:
                    return self._get_closest(arr[mid - 1], arr[mid], target)
                j = mid
            else:
                if mid < n - 1 and target < arr[mid + 1]:
                    return self._get_closest(arr[mid], arr[mid + 1], target)
                i = mid + 1
        return arr[mid]

    def _get_closest(self,val1, val2, target):
        if target-val1 >= val2-target:
            return val2
        else:
            return val1

