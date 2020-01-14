import torch
import os,sys,glob
from abc import ABC, abstractmethod
from TSX.utils import train_reconstruction, test_reconstruction, train_model, train_model_rt, train_model_rt_rg, test, test_model_rt_rg, logistic, test_model_rt, test, test_cond
from TSX.models import EncoderRNN, RiskPredictor, LR, RnnVAE
from TSX.generator import FeatureGenerator, train_joint_feature_generator, train_feature_generator, CarryForwardGenerator, DLMGenerator, JointFeatureGenerator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.image as mpimg
import json
import time
from math import log
from statistics import mean

import lime
import lime.lime_tabular
from TSX.temperature_scaling import ModelWithTemperature

#generic plot configs
line_styles_map=['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--','-.',':']
marker_styles_map=['o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h']

#mimic plot configs
xkcd_colors = mcolors.XKCD_COLORS
color_map = [list(xkcd_colors.keys())[k] for k in
             np.random.choice(range(len(xkcd_colors)), 28, replace=False)]
color_map = ['#00998F', '#C20088', '#0075DC', '#E0FF66', '#4C005C', '#191919', '#FF0010', '#2BCE48', '#FFCC99', '#808080',
             '#740AFF', '#8F7C00', '#9DCC00', '#F0A3FF', '#94FFB5', '#FFA405', '#FFA8BB', '#426600', '#005C31', '#5EF1F2',
             '#993F00', '#990000', '#003380', '#990000', '#FFFF80', '#FF5005', '#FFFF00','#FF0010', '#FFCC99']

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']

feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

#simulation plot configs
feature_map_simulation = ['feature 0', 'feature 1', 'feature 2']

feature_map = {'mimic':feature_map_mimic, 'simulation': feature_map_simulation}

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
        train_start_time = time.time()
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
        print("Generator training time = ", time.time()-train_start_time)


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
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor',simulation=False,data='mimic', model='RNN'):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader, data=data)
        if model=='RNN':
            self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=False,data=data)
        elif model=='LR':
            self.model = LR(feature_size)
        self.model_type = model
        self.experiment = experiment
        self.simulation = simulation
        self.data = data
        self.ckpt_path='./ckpt/' + self.data

    def run(self, train,n_epochs, **kwargs):
        if train:
            self.train(n_epochs=n_epochs, learn_rt=self.data!='mimic')
        else:
            path = './ckpt/' + self.data + '/' + str(self.experiment) + '_' + self.model_type + '.pt'
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path))

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
                                            self.experiment+'_'+self.model_type, data=self.data)
            # Evaluate performance on held-out test set
            _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            print('\nFinal performance on held out test set ===> AUC: ', auc_test)
        else:
            if self.data=='simulation':
                print('training rt')
                train_model_rt(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                            self.experiment+'_'+self.model_type, data=self.data)
                # Evaluate performance on held-out test set
                _, _, _, auc, correct_label = test_model_rt(self.model, self.valid_loader)
                print('Precalibration auc: ', auc)
                self.model = ModelWithTemperature(self.model)
                self.model.set_temperature(self.valid_loader)
                self.model.eval()
                _, _, _, auc, correct_label = test_model_rt(self.model, self.valid_loader)
                print('\nPost calibration AUC: ', auc)
                print('\nFinal performance on held out test set ===> AUC: ', auc)

            else:
                #only for ghg data
                train_model_rt_rg(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                               self.experiment,data=self.data)


class BaselineExplainer(Experiment):
    """ Baseline explainability methods
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, data_class, experiment='baseline_explainer', data='mimic', baseline_method='lime',**kwargs):
        super(BaselineExplainer, self).__init__(train_loader, valid_loader, test_loader, data=data)
        self.experiment = experiment
        self.data_class = data_class
        self.ckpt_path = os.path.join('./ckpt/', self.data)
        self.baseline_method = baseline_method
        self.input_size = feature_size
        self.spike_data = kwargs['spike_data'] if 'spike_data' in kwargs.keys() else False
        self.learned_risk = True
        if data == 'mimic':
            self.timeseries_feature_size = len(feature_map_mimic)
        else:
            self.timeseries_feature_size = feature_size

        # Build the risk predictor and load checkpoint
        with open('config.json') as config_file:
            if not self.spike_data:
                configs = json.load(config_file)[data]['risk_predictor']
            else:
                configs = json.load(config_file)['simulation_spike']['risk_predictor']
        if self.data == 'simulation':
            if not self.learned_risk:
                self.risk_predictor = lambda signal,t:logistic(2.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t] - 1))
            else:
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
                                                 rnn=configs['rnn_type'], regres=True, data=data)
                # self.risk_predictor = EncoderRNN(configs['feature_size'], hidden_size=configs['encoding_size'],
                #                                  rnn=configs['rnn_type'], regres=True, return_all=False, data=data)
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
        #self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor.pt')))
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

    def run(self, train, n_epochs=60, samples_to_analyze=MIMIC_TEST_SAMPLES):
        self.train(n_epochs=n_epochs, learn_rt=self.data=='ghg')
        testset = list(self.test_loader.dataset)
        test_signals = torch.stack(([x[0] for x in testset])).to(self.device)
        matrix_test_dataset = test_signals.cpu().numpy()
        # importance = np.zeros((len(samples_to_analyze),test_signals.shape[1], test_signals.shape[2]))
        explanation = []
        exec_time = []
        for test_sample_ind, test_sample in enumerate(samples_to_analyze):
            exp_sample=[]
            lime_start = time.time()
            for tttt in range(matrix_test_dataset.shape[2]):
                exp = self.explainer.explain_instance(np.mean(matrix_test_dataset[test_sample,:,:tttt+1],axis=1), self.predictor_wrapper, labels=['feature '+ str(i) for i in range(len(self.feature_map))], top_labels=2)
                #print("Most important features for sample %d: " % (test_sample), exp.as_list())
                exp_sample.append(exp.as_list())
            explanation.append(exp_sample)
            exec_time.append(time.time()-lime_start)
        exec_time = np.array(exec_time)
        print("Execution time of lime for subject %d= %.3f +/- %.3f"%(test_sample, np.mean(exec_time), np.std(np.array(exec_time))))
        return explanation
            # for t in range(test_signals.shape[-1]):
            #     exp = self.explainer.explain_instance(test_signals[test_sample, :, t], self.predictor_wrapper, top_labels=2)
            #     print("Most important features for sample %d: "%(test_sample), exp.as_list())
            #     for f in range(test_signals.shape[1]):
            #         imp = exp.as_list(f)
            #         importance[test_sample_ind,f,t] = imp
        # print(importance[0,2,10:13])

    def train(self, n_epochs, learn_rt=False):
        trainset = list(self.train_loader.dataset)
        signals = torch.stack(([x[0] for x in trainset])).to(self.device)
        matrix_train_dataset = signals.mean(dim=2).cpu().numpy()
        if self.baseline_method == 'lime':
            #self.explainer = lime.lime_tabular.LimeTabularExplainer(matrix_train_dataset, discretize_continuous=True)
            self.explainer = lime.lime_tabular.LimeTabularExplainer(matrix_train_dataset, feature_names=self.feature_map+['gender', 'age', 'ethnicity', 'first_icu_stay'], discretize_continuous=True)


class FeatureGeneratorExplainer(Experiment):
    """ Experiment for generating feature importance over time using a generative model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, patient_data, generator_hidden_size=80, prediction_size=1, historical=False, generator_type='RNN_generator', experiment='feature_generator_explainer', data='mimic', conditional=True,**kwargs):
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
        elif 'joint_RNN_generator' in self.generator_type :
            self.generator = JointFeatureGenerator(feature_size, data=data).to(self.device) # TODO setup the right encoding size
        elif self.generator_type == 'dlm_joint_generator':
            self.generator = DLMGenerator(feature_size).to(self.device) # TODO setup the right encoding size
        else:
            raise RuntimeError('Undefined generator!')

        if data == 'mimic':
            self.timeseries_feature_size = len(feature_map_mimic)
        else:
            self.timeseries_feature_size = feature_size

        self.feature_size = feature_size
        self.input_size = feature_size
        self.patient_data = patient_data
        self.experiment = experiment
        self.historical = historical
        self.simulation = self.data=='simulation'
        self.spike_data=kwargs['spike_data'] if "spike_data" in kwargs.keys() else False
        self.prediction_size = prediction_size
        self.generator_hidden_size = generator_hidden_size
        self.ckpt_path = os.path.join('./ckpt/', self.data)
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
                if self.spike_data:
                    self.risk_predictor = EncoderRNN(feature_size,hidden_size=50,rnn='GRU',regres=True, return_all=False,data=data)
                else:
                    self.risk_predictor = EncoderRNN(feature_size,hidden_size=100,rnn='GRU',regres=True, return_all=False,data=data)
            self.feature_map = feature_map_simulation
        else:
            if self.data=='mimic':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=150, rnn='GRU', regres=True,data=data)
                self.feature_map = feature_map_mimic
                self.risk_predictor = self.risk_predictor.to(self.device)
            elif self.data=='ghg':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=500, rnn='LSTM', regres=True,data=data)
                self.feature_map = feature_map_ghg
                self.risk_predictor = self.risk_predictor.to(self.device)

    def run(self, train,n_epochs, samples_to_analyze, plot=True, sanity_check=None):
        """ Run feature generator experiment
        :param train: (boolean) If True, train the generators, if False, use saved checkpoints
        """
        testset = list(self.test_loader.dataset)
        if train and self.generator_type!='carry_forward_generator':
           self.train(n_features=self.timeseries_feature_size, n_epochs=n_epochs)
           return 0
        else:
            # ckpt_path = os.path.join('./ckpt',self.data)
            if self.historical:
                check_path = glob.glob(os.path.join(self.ckpt_path,'*_generator.pt'))[0]
            else:
                check_path = glob.glob(os.path.join(self.ckpt_path,'*_generator.pt'))[0]

            if not os.path.exists(check_path):
                raise RuntimeError('No saved checkpoint for this model')

            else:
                if not self.data=='simulation':
                    if sanity_check == "randomized_param": # Do not load learnt parameters if doing experiment with randomized parameters
                        self.risk_predictor.load_state_dict(
                            torch.load(os.path.join(self.ckpt_path, 'risk_predictor_RNN.pt')))
                        for param in self.risk_predictor.parameters():
                            params = param.data.cpu().numpy().reshape(-1)
                            np.random.shuffle(params)
                            param.data = torch.Tensor(params.reshape(param.data.shape))
                            # param.data = torch.rand(param.data.size())
                    elif sanity_check == "randomized_model":
                        self.risk_predictor = LR(self.feature_size)
                        self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor_LR.pt')))
                    elif sanity_check == "randomized_data":
                        trainset = list(self.train_loader.dataset)
                        shuffled_labels = torch.stack([sample[1] for sample in trainset])
                        r = torch.randperm(len(shuffled_labels))
                        shuffled_labels = shuffled_labels[r]
                        signals = torch.stack([sample[0] for sample in trainset])
                        shuffled_trainset = torch.utils.data.TensorDataset(signals, shuffled_labels)
                        shuffled_trainloader = torch.utils.data.DataLoader(shuffled_trainset, batch_size=100)
                        self.risk_predictor.train()
                        print("Training model on shuffled data ...")
                        opt = torch.optim.Adam(self.risk_predictor.parameters(), lr=0.0001, weight_decay=1e-3)
                        train_model(self.risk_predictor, shuffled_trainloader, self.valid_loader, opt, n_epochs, self.device, self.experiment, data=self.data)
                    else:
                        self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path,'risk_predictor_RNN.pt')))

                    # Calibrate model
                    # self.risk_predictor = self.risk_predictor.to(self.device)
                    # _, _, auc, correct_label, _ = test(self.test_loader, self.risk_predictor, self.device)
                    # print('Precalibration auc: ', auc)
                    # self.risk_predictor = ModelWithTemperature(self.risk_predictor)
                    # self.risk_predictor.set_temperature(self.valid_loader)
                        self.risk_predictor.to(self.device)
                    self.risk_predictor.eval()
                    _, _, auc, correct_label, _ = test(self.test_loader, self.risk_predictor, self.device)
                    # print('Postcalibration auc: ', auc)


                else: #simulated data
                    if self.learned_risk:
                        if sanity_check == "randomized_param":  # Do not load learnt parameters if doing experiment with randomized parameters
                            self.risk_predictor.load_state_dict(
                                torch.load(os.path.join(self.ckpt_path, 'risk_predictor_RNN.pt')))
                            self.risk_predictor = self.risk_predictor.to(self.device)
                            self.risk_predictor.eval()
                            _, _, precision, auc, correct_label = test_model_rt(self.risk_predictor, self.test_loader)
                            print("\nRisk predictor model AUC before randomization:%.2f" % (auc))
                            for param in self.risk_predictor.parameters():
                                # params = param.data.cpu().numpy().reshape(-1)
                                # np.random.shuffle(params)
                                # param.data = torch.Tensor(params.reshape(param.data.shape))
                                param.data = torch.randn(param.data.size())
                        elif sanity_check == "randomized_model":
                            self.risk_predictor = LR(self.feature_size)
                            self.risk_predictor.load_state_dict(
                                torch.load(os.path.join(self.ckpt_path, 'risk_predictor_LR.pt')))
                        elif sanity_check == "randomized_data":
                            trainset = list(self.train_loader.dataset)
                            shuffled_labels = np.random.randint(2, size=(len(trainset),trainset[0][1].shape[0]))
                            signals = torch.stack([sample[0] for sample in trainset])
                            shuffled_trainset = torch.utils.data.TensorDataset(signals, torch.Tensor(shuffled_labels))
                            shuffled_trainloader = torch.utils.data.DataLoader(shuffled_trainset, batch_size=100)
                            self.risk_predictor.train()
                            opt = torch.optim.Adam(self.risk_predictor.parameters(), lr=0.001, weight_decay=0)
                            print("Training model on shuffled data ...")
                            train_model_rt(self.risk_predictor, shuffled_trainloader, self.valid_loader, opt, n_epochs,
                                           self.device, self.experiment, data=self.data)
                        else:
                            self.risk_predictor.load_state_dict(
                                torch.load(os.path.join(self.ckpt_path, 'risk_predictor_RNN.pt')))

                        # Calibrate model
                        # self.risk_predictor = self.risk_predictor.to(self.device)
                        # _, _, _,auc,  _ = test_model_rt(self.risk_predictor, self.test_loader)
                        # print('Precalibration auc: ', auc)
                        # self.risk_predictor = ModelWithTemperature(self.risk_predictor)
                        # self.risk_predictor.set_temperature(self.valid_loader)
                        self.risk_predictor.to(self.device)
                        self.risk_predictor.eval()
                        _, _, _, auc,  _ = test_model_rt(self.risk_predictor, self.test_loader)
                        # print('Postcalibration auc: ', auc)


                        # self.risk_predictor.eval()
                        # _, _, precision, auc, correct_label = test_model_rt(self.risk_predictor, self.test_loader)

            print("\nRisk predictor model AUC:%.2f"%(auc))

            if self.data=='simulation':
                if self.spike_data==True:
                    with open(os.path.join('./data_generator/data/simulated_data/thresholds_test.pkl'), 'rb') as f:
                        th = pkl.load(f)

                    with open(os.path.join('./data_generator/data/simulated_data/gt_test.pkl'), 'rb') as f:
                        gt_importance = pkl.load(f)#Type dmesg and check the last few lines of output. If the disc or the connection to it is failing, it'll be noted there.load(f)
                else:
                    with open(os.path.join('./data/simulated_data/state_dataset_importance_test.pkl'),'rb') as f:
                        gt_importance = pkl.load(f)


                #For simulated data this is the last entry - end of 48 hours that's the actual outcome
                label = np.array([x[1][-1] for x in testset])
                #print(label)
                high_risk = np.where(label==1)[0]
                if len(samples_to_analyze)==0:
                    samples_to_analyze = np.random.choice(range(len(label)), len(label), replace=False)
                # samples_to_analyse = [101, 48, 88, 192, 143, 166, 18, 58, 172, 132]
            else:
                gt_importance = None
                # if self.data=='mimic':
                    # samples_to_analyse = MIMIC_TEST_SAMPLES
                if self.data=='ghg':
                    label = np.array([x[1][-1] for x in testset])
                    high_risk = np.arange(label.shape[0])
                    samples_to_analyze = np.random.choice(high_risk, len(high_risk), replace=False)

            ## Sensitivity analysis as a baseline
            signal = torch.stack([testset[sample][0] for sample in samples_to_analyze])

            #Some setting up for ghg data
            if self.data=='ghg':
                label_tch = torch.stack([testset[sample][1] for sample in samples_to_analyze])
                signal_scaled = self.patient_data.scaler_x.inverse_transform(np.reshape(signal.cpu().detach().numpy(),[len(samples_to_analyze),-1]))
                signal_scaled = np.reshape(signal_scaled,signal.shape)
                #label_scaled = self.patient_data.scaler_y.inverse_transform(np.reshape(label_tch.cpu().detach().numpy(),[len(samples_to_analyze),-1]))
                #label_scaled = np.reshape(label_scaled,label_tch.shape)
                label_scaled = label_tch.cpu().detach().numpy()
                #tvec = [int(x) for x in np.linspace(1,signal.shape[2]-1)]
                tvec = list(range(1,signal.shape[2]+1,50))
            else:
                tvec = list(range(1,signal.shape[2]+1))
                signal_scaled = signal

            nt = len(tvec)
            sensitivity_analysis = np.zeros((signal.shape))
            sensitivity_start = time.time()
            if not self.data=='simulation':
                if self.data=='mimic' or self.data=='ghg':
                    self.risk_predictor.train()
                    for t_ind,t in enumerate(tvec):
                        signal_t = torch.Tensor(signal[:,:,:t]).to(self.device).requires_grad_()
                        out = self.risk_predictor(signal_t)
                        for s in range(len(samples_to_analyze)):
                            out[s].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:,t_ind].cpu().detach().numpy()
                        signal_t.grad.data.zero_()
                self.risk_predictor.eval()
            else:
                if not self.learned_risk:
                    grad_out = []
                    for kk,i in enumerate(samples_to_analyze):
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
                        for s in range(len(samples_to_analyze)):
                            out[s].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:, t_ind].cpu().detach().numpy()#[:,0]
                        signal_t.grad.data.zero_()
                    self.risk_predictor.eval()
            print('Execution time of sensitivity = %.3f'%((time.time()-sensitivity_start)/float(len(samples_to_analyze))))
            print('\n********** Visualizing a few samples **********')
            all_FFC_importance = []
            all_FO_importance = []
            all_AFO_importance = []
            all_lime_importance = []

            # self.risk_predictor.load_state_dict(torch.load(os.path.join('./ckpt',self.data,'risk_predictor.pt')))
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
                lime_exp = BaselineExplainer(self.train_loader, self.valid_loader, self.test_loader,
                                             self.feature_size, data_class=self.patient_data,
                                             data=self.data, baseline_method='lime',spike_data=self.spike_data)
                importance_labels = {}
                for sub_ind, sample_ID in enumerate(samples_to_analyze):
                    print('Fetching importance results for sample %d' % sample_ID)

                    lime_imp = lime_exp.run(train=train, n_epochs=n_epochs, samples_to_analyze=[sample_ID])
                    all_lime_importance.append(lime_imp)

                    top_FCC, importance, top_occ, importance_occ, top_occ_aug, importance_occ_aug, top_SA, importance_SA = self.plot_baseline(
                        sample_ID, signals_to_analyze, sensitivity_analysis[sub_ind, :, :], data=self.data, plot=plot, 
                        gt_importance_subj=gt_importance[sample_ID] if self.data == 'simulation' else None,lime_imp=lime_imp,tvec=tvec)
                    all_FFC_importance.append(importance)
                    all_AFO_importance.append(importance_occ_aug)
                    all_FO_importance.append(importance_occ)

                    top_signals = 4

                    FFC = []
                    for ind, sig in top_FCC[0:top_signals]:
                        ref_ind = signals_to_analyze[ind]
                        imp_t = importance[ind, :]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        FFC.append((i_max, t_max, max_val))
                    importance_labels.update({'FFC':FFC})

                    FO = []
                    for ind, sig in top_occ[0:top_signals]:
                        ref_ind = signals_to_analyze[ind]
                        imp_t = importance_occ[ind, :]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        FO.append((i_max, t_max, max_val))
                    importance_labels.update({'FO': FO})

                    SA = []
                    for ind, sig in top_SA[0:top_signals]:
                        ref_ind = signals_to_analyze[ind]
                        imp_t = importance_SA[ind, 1:]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        SA.append((i_max, t_max, max_val))
                    importance_labels.update({'SA': SA})

                    AFO = []
                    for ind, sig in top_occ_aug[0:top_signals]:
                        ref_ind = signals_to_analyze[ind]
                        imp_t = importance_occ_aug[ind, :]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        AFO.append((i_max, t_max,max_val))
                    importance_labels.update({'AFO': AFO})

                   #importance_labels.update({'lime': lime_imp})

                    with open('./examples/%s/baseline_importance_sample_%d.json'%(self.data, sample_ID),'w') as f:
                        json.dump(importance_labels, f)
        return np.array(all_FFC_importance), np.array(all_AFO_importance), np.array(all_FO_importance), np.array(all_lime_importance), sensitivity_analysis

    def plot_baseline(self, subject, signals_to_analyze, sensitivity_analysis_importance, retain_style=False, plot=True,  n_important_features=3,data='mimic',gt_importance_subj=None,lime_imp=None,tvec=None):
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
        if data=='mimic':
            print('Did this patient die? ', {1: 'yes', 0: 'no'}[label_o.item()])
        #if tvec is None:
        tvec = range(1,signals.shape[1])

        importance = np.zeros((self.timeseries_feature_size, len(tvec)))
        mean_predicted_risk = np.zeros((self.timeseries_feature_size, len(tvec)))
        std_predicted_risk = np.zeros((self.timeseries_feature_size, len(tvec)))
        importance_occ = np.zeros((self.timeseries_feature_size, len(tvec)))
        std_predicted_risk_occ = np.zeros((self.timeseries_feature_size, len(tvec)))
        importance_occ_aug = np.zeros((self.timeseries_feature_size, len(tvec)))
        std_predicted_risk_occ_aug = np.zeros((self.timeseries_feature_size, len(tvec)))
        max_imp_FCC = []
        max_imp_occ = []
        max_imp_occ_aug = []
        max_imp_sen = []
        AFO_exe_time = []
        FO_exe_time = []
        FFC_exe_time = []
        # TODO for joint models we don't need to iterate over separate signals. One counterfactual generation is enough!
        for i, sig_ind in enumerate(signals_to_analyze):

            if not self.generator_type=='carry_forward_generator':
                if 'joint' in self.generator_type:
                    self.generator.load_state_dict(
                        torch.load(os.path.join('./ckpt/%s/%s.pt'%(self.data, self.generator_type))))
                else:
                    if data=='mimic':
                        self.generator.load_state_dict(
                        torch.load(os.path.join('./ckpt',data,'%s_%s.pt' % (feature_map_mimic[sig_ind], self.generator_type))))
                    elif data=='simulation':
                        self.generator.load_state_dict(
                        torch.load(os.path.join('./ckpt',data,'%s_%s.pt' % (str(sig_ind), self.generator_type))))
                    elif data=='ghg':
                        self.generator.load_state_dict(
                        torch.load(os.path.join('./ckpt',data,'%s_%s.pt' % (str(sig_ind),self.generator_type))))

            t0 = time.time()
            label, importance[i, :] = self._get_feature_importance_FFC( signals, sig_ind=sig_ind, n_samples=10, learned_risk=self.learned_risk)
            # label, importance[i, :], mean_predicted_risk[i, :], std_predicted_risk[i, :] = self._get_feature_importance(
            #     signals, sig_ind=sig_ind, n_samples=10, mode='generator', learned_risk=self.learned_risk)#,tvec=tvec)
            t1 = time.time()
            FFC_exe_time.append(t1-t0)

            t0 = time.time()
            _, importance_occ[i, :], _, std_predicted_risk_occ[i, :] = self._get_feature_importance(signals,
                                                                                                    sig_ind=sig_ind,
                                                                                                    n_samples=10,
                                                                                                    mode="feature_occlusion",
                                                                                                    learned_risk=self.learned_risk,tvec=tvec)
            FO_exe_time.append(time.time()-t0)

            t0 = time.time()
            _, importance_occ_aug[i, :], _, std_predicted_risk_occ_aug[i, :] = self._get_feature_importance(signals,
                                                                                                            sig_ind=sig_ind,
                                                                                                            n_samples=10,
                                                                                                            mode='augmented_feature_occlusion',
                                                                                                            learned_risk=self.learned_risk,tvec=tvec)
            AFO_exe_time.append(time.time()-t0)
            max_imp_FCC.append((i, max(importance[i, :])))
            max_imp_occ.append((i, max(importance_occ[i, :])))
            max_imp_occ_aug.append((i, max(importance_occ_aug[i, :])))
            max_imp_sen.append((i, max(sensitivity_analysis_importance[i, :])))

        print('Execution time of FFC for subject %d = %.3f +/- %.3f'%(subject, np.mean(np.array(FFC_exe_time)), np.std(np.array(FFC_exe_time))) )
        print('Execution time of AFO for subject %d = %.3f +/- %.3f' % (subject, np.mean(np.array(AFO_exe_time)), np.std(np.array(AFO_exe_time))))
        print('Execution time of FO for subject %d = %.3f +/- %.3f' % (subject, np.mean(np.array(FO_exe_time)), np.std(np.array(FO_exe_time))))
        if self.spike_data:
            datafile='simulation_spike'
        # else:
        #     datafile = data
        # lime_exp = BaselineExplainer(self.train_loader, self.valid_loader, self.test_loader,
        #                                      self.feature_size, data_class=self.patient_data,
        #                                      data=self.data, baseline_method='lime')
        # lime_imp = lime_exp.run(train=True, n_epochs=100, samples_to_analyze=[subject])
        with open(os.path.join('/scratch/gobi1/sana/TSX_results/',data,'results_'+str(subject)+'.pkl'), 'wb') as f:
        #with open(os.path.join('/scratch/gobi1/sana/TSX_results/simulation/', 'results_' + str(subject) + '.pkl'), 'wb') as f:
        #with open(os.path.join('./examples',data,'results_'+str(subject)+'.pkl'),'wb') as f:
            pkl.dump({'FFC': {'imp':importance,'std':std_predicted_risk}, 'Suresh_et_al':{'imp':importance_occ,'std':std_predicted_risk_occ}, 'AFO': {'imp':importance_occ_aug,'std': std_predicted_risk_occ_aug}, 'Sens': {'imp': sensitivity_analysis_importance,'std':[]}, 'lime':{'imp':lime_imp, 'std':[]},  'gt':gt_importance_subj},f,protocol=pkl.HIGHEST_PROTOCOL)
        ## Plot heatmaps
        import seaborn as sns
        sns.set()
        all_importances = np.stack([importance, importance_occ, importance_occ_aug, abs(sensitivity_analysis_importance[:len(importance),1:])], axis=0)
        all_importance_labels = ['FFC', 'FO', 'AFO', 'Sens']
        for imp_plot_ind in range(4):
            heatmap_fig = plt.figure(figsize=(15, 1) if data=='simulation' else (16,9))
            plt.yticks(rotation=0)
            imp_plot = sns.heatmap(all_importances[imp_plot_ind], yticklabels=feature_map[data], square=True if data=='mimic' else False)#, vmin=0, vmax=1)
            heatmap_fig.savefig(os.path.join('/scratch/gobi1/sana/TSX_results/',data, 'heatmap_'+str(subject)+'_'+all_importance_labels[imp_plot_ind]+'.pdf'))
        if data=='simulation':
                heatmap_gt = plt.figure(figsize=(20, 1))
                plt.yticks(rotation=0)
                imp_plot = sns.heatmap(gt_importance_subj, yticklabels=feature_map[data])
                heatmap_gt.savefig(os.path.join('/scratch/gobi1/sana/TSX_results/',data, 'heatmap_'+str(subject)+'_ground_truth.pdf'))
        if not plot:
            return max_imp_FCC, importance, max_imp_occ, importance_occ, max_imp_occ_aug, importance_occ_aug, max_imp_sen, sensitivity_analysis_importance


        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
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
        n_output_importance = 5
        import heapq
        from TSX.generator_baselines import find_true_gen_importance

        # TODO Remove first important assignments for FFC, becuase poor quality of the generator results in different scaling
        # FCC
        for ind, sig in max_imp_FCC[0:n_feats_to_plot]:
            ref_ind = signals_to_analyze[ind]
            if ref_ind not in important_signals:
                important_signals.append(ref_ind)
            c = color_map[ref_ind]
            ax2.errorbar(t, importance[ind, :], yerr=std_predicted_risk[ind, :],
                         marker=markers[list(important_signals).index(ref_ind) % len(markers)],
                         markersize=9, markeredgecolor='k', linewidth=3, elinewidth=1,
                         linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)], color=c,
                         label='%s' % (self.feature_map[ref_ind]))
        max_ts = importance.reshape(-1).argsort()[-1*n_output_importance:][::-1]
        max_ts = [(t//(importance.shape[-1]), t%(importance.shape[-1]) ) for t in max_ts]
        for imp_f, imp_t in max_ts:
            imp_f, imp_t = int(imp_f), int(imp_t)
            c = color_map[imp_f]
            ax2.axvspan(imp_t-1, imp_t+1, facecolor=c, alpha=0.3, hatch='/', label='%s score: %.3f' % (self.feature_map[imp_f], importance[imp_f, imp_t]))
            # ax2.annotate('%.2f'%importance[imp_f, imp_t], (imp_t-.5, importance[imp_f, imp_t]+.1), fontsize=26, fontweight='bold')
        pos = ax2.get_position() # get the original position
        ax2.text(pos.x0 - .2, pos.y0 +.1, 'FFC', transform=ax2.transAxes, fontsize=46, fontweight='bold', va='top', bbox=props)


        # if self.data=='simulation':
        #     gt_importance_subj = gt_importance[subject,:]
        if not gt_importance_subj is None:
            # Shade the state on simulation data plots
            if gt_importance_subj.shape[0]==3:
                gt_importance_subj =gt_importance_subj.transpose(1,0)
            if not self.spike_data:
                for ax in [ax1]:#, ax2, ax3, ax4, ax5]:
                    prev_color = 'g' if np.argmax(gt_importance_subj[:, 1])<np.argmax(gt_importance_subj[:, 2]) else 'y'
                    for ttt in range(1,len(t)+1):
                        if gt_importance_subj[ttt, 1]==1:
                            ax.axvspan(ttt-1,ttt,facecolor='g',alpha=0.3)
                            prev_color = 'g'
                        elif gt_importance_subj[ttt, 2]==1:
                            ax.axvspan(ttt-1,ttt,facecolor='y',alpha=0.3)
                            prev_color = 'y'
                        elif not prev_color is None:
                            ax.axvspan(ttt - 1, ttt, facecolor=prev_color, alpha=0.3)
            else:
                for ttt in range(1,len(t)+1):
                    if gt_importance_subj[ttt]==1:
                        ax1.axvspan(ttt-1,ttt,facecolor='g',alpha=0.3)

                for ttt in  range(1,len(t)+1):
                    if gt_importance_subj[ttt]==1:
                        ax2.axvspan(ttt-1,ttt,facecolor='g',alpha=0.3)

                for ttt in range(1,len(t)+1):
                    if gt_importance_subj[ttt]==1:
                        ax3.axvspan(ttt-1,ttt,facecolor='g',alpha=0.3)

                for ttt in range(1,len(t)+1):
                    if gt_importance_subj[ttt]==1:
                        ax4.axvspan(ttt-1,ttt,facecolor='g',alpha=0.3)


        # Augmented feature occlusion
        for ind, sig in max_imp_occ_aug[0:n_feats_to_plot]:
            ref_ind = signals_to_analyze[ind]
            if ref_ind not in important_signals:
                important_signals.append(ref_ind)
            c = color_map[ref_ind]
            ax3.errorbar(t, importance_occ_aug[ind, :], yerr=std_predicted_risk_occ_aug[ind, :],
                         marker=markers[list(important_signals).index(ref_ind) % len(markers)], linewidth=3, elinewidth=1,
                         linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)],
                         markersize=9, markeredgecolor='k', color=c, label='%s' % (self.feature_map[ref_ind]))
        max_ts = importance_occ_aug.reshape(-1).argsort()[-1 * n_output_importance:][::-1]
        max_ts = [(t // (importance_occ_aug.shape[-1]), t % (importance_occ_aug.shape[-1])) for t in max_ts]
        for imp_f, imp_t in max_ts:
            imp_f, imp_t = int(imp_f), int(imp_t)
            c = color_map[imp_f]
            ax3.axvspan(imp_t - 1, imp_t + 1, facecolor=c, alpha=0.3, hatch='/',
                            label='%s score: %.3f' % (self.feature_map[imp_f], importance_occ_aug[imp_f, imp_t]))
            # ax3.annotate('%.2f' % importance_occ_aug[imp_f, imp_t], (imp_t-.5, importance_occ_aug[imp_f, imp_t] + .1), fontsize=26, fontweight='bold')
        pos = ax3.get_position() # get the original position
        ax3.text(pos.x0 - .2, pos.y0 +.1, 'AFO', transform=ax3.transAxes, fontsize=46, fontweight='bold', va='top', bbox=props)

        # Feature occlusion
        for ind, sig in max_imp_occ[0:n_feats_to_plot]:
            ref_ind = signals_to_analyze[ind]
            if ref_ind not in important_signals:
                important_signals.append(ref_ind)
            c = color_map[ref_ind]
            ax4.errorbar(t, importance_occ[ind, :], yerr=std_predicted_risk_occ[ind, :],
                         marker=markers[list(important_signals).index(ref_ind) % len(markers)],
                         linewidth=3, elinewidth=1, linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)],
                         markersize=9, markeredgecolor='k',
                         color=c, label='%s' % (self.feature_map[ref_ind]))

        max_ts = importance_occ.reshape(-1).argsort()[-1 * n_output_importance:][::-1]
        max_ts = [(t // (importance_occ.shape[-1]), t % (importance_occ.shape[-1])) for t in max_ts]
        for imp_f, imp_t in max_ts:
            imp_f, imp_t = int(imp_f), int(imp_t)
            c = color_map[imp_f]
            ax4.axvspan(imp_t - 1, imp_t + 1, facecolor=c, alpha=0.3, hatch='/',
                            label='%s score: %.3f' % (self.feature_map[imp_f], importance_occ[imp_f, imp_t]))
            # ax4.annotate('%.2f' % importance_occ[imp_f, imp_t], (imp_t-.5, importance_occ[imp_f, imp_t] + .1), fontsize=26, fontweight='bold')
        pos = ax4.get_position() # get the original position
        ax4.text(pos.x0 - .2, pos.y0 +.25, 'FO', transform=ax4.transAxes, fontsize=46, fontweight='bold', va='top', bbox=props)

        # Sensitivity analysis
        for ind, sig in max_imp_sen[0:n_feats_to_plot]:
            ref_ind = signals_to_analyze[ind]
            if ref_ind not in important_signals:
                important_signals.append(ref_ind)
            c = color_map[ref_ind]
            ax5.plot(abs(sensitivity_analysis_importance[ind, 1:]), linewidth=3,
                     linestyle=l_style[list(important_signals).index(ref_ind) % len(l_style)],
                     color=c, label='%s' % (self.feature_map[ref_ind]))
        if self.data=='mimic':
            sensitivity_analysis_importance = sensitivity_analysis_importance[:self.timeseries_feature_size, 1:]
        else:
            sensitivity_analysis_importance = sensitivity_analysis_importance[:,1:]
        max_ts = sensitivity_analysis_importance.reshape(-1).argsort()[-1 * n_output_importance:][::-1]
        max_ts = [(t // (sensitivity_analysis_importance.shape[-1]), t % (sensitivity_analysis_importance.shape[-1])) for t in max_ts]
        for imp_f, imp_t in max_ts:
            imp_f, imp_t = int(imp_f), int(imp_t)
            c = color_map[imp_f]
            ax5.axvspan(imp_t - 1, imp_t + 1, facecolor=c, alpha=0.3, hatch='/',
                            label='%s score: %.3f' % (self.feature_map[imp_f], sensitivity_analysis_importance[imp_f, imp_t]))
            # ax5.annotate('%.2f' % sensitivity_analysis_importance[imp_f, imp_t], (imp_t-.5, sensitivity_analysis_importance[imp_f, imp_t] + .1), fontsize=26, fontweight='bold')
        pos = ax5.get_position() # get the original position 
        ax5.text(pos.x0 - .2, pos.y0 +.5, 'SA', transform=ax5.transAxes, fontsize=46, fontweight='bold', va='top', bbox=props)

        # Plot the original signals
        important_signals = np.unique(important_signals)
        max_plot = (torch.max(torch.abs(signals[important_signals,:]))).item()
        # max_plot = max( max(signals.cpu().numpy()[important_signals,:]). abs(min(signals.cpu().numpy()[important_signals,:])))
        for i, ref_ind in enumerate(important_signals):
            c = color_map[ref_ind]
            #ax1.plot(np.array(signals[ref_ind, 1:] / max(abs(signals[ref_ind, 1:]))), linewidth=3,
            #         linestyle=l_style[i % len(l_style)], color=c,
            #         label='%s' % (self.feature_map[ref_ind]))
            if self.data=='mimic':
                ax1.plot(np.array(signals[ref_ind, 1:])/max_plot, linewidth=3, linestyle=l_style[i % len(l_style)], color=c, label='%s' % (self.feature_map[ref_ind]))
            else:
                ax1.plot(np.array(signals[ref_ind, 1:]), linewidth=3, linestyle=l_style[i % len(l_style)],
                         color=c, label='%s' % (self.feature_map[ref_ind]))

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.grid()
            ax.tick_params(axis='both', labelsize=36)
            if ax!=ax1:
                ax.set_ylabel('importance', fontweight='bold', fontsize=32)
                if self.data == 'simulation' and ax!=ax5:
                    ax.set_ylim(top=.8)


        ax1.plot(np.array(label), '--', linewidth=6, label='Risk score', c='black')

        ax1.set_title('Time series signals and Model\'s predicted risk', fontweight='bold', fontsize=34)
        # ax2.set_title('FFC', fontweight='bold', fontsize=34) # TODO change the title depending on the type of generator is being used
        # ax3.set_title('AFO', fontweight='bold', fontsize=34)
        # ax4.set_title('FO', fontweight='bold', fontsize=34)
        # ax5.set_title('Sensitivity analysis', fontweight='bold', fontsize=34)
        ax5.set_xlabel('time', fontweight='bold', fontsize=32)
        ax1.set_ylabel('signal value', fontweight='bold',fontsize=32)

        f.set_figheight(25)
        f.set_figwidth(60)
        #plt.savefig(os.path.join('./examples',data,'feature_%d_%s.pdf' %(subject, self.generator_type)), dpi=300, orientation='landscape')#,
                    #bbox_inches='tight')
        plt.savefig(os.path.join('/scratch/gobi1/sana/TSX_results',data,'feature_%d_%s.pdf' %(subject, self.generator_type)), dpi=300, orientation='landscape')
        fig_legend = plt.figure(figsize=(13, 1.2))
        handles, labels = ax1.get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
        #fig_legend.savefig(os.path.join('./examples', data, 'legend_%d_%s.pdf' %(subject, self.generator_type)), dpi=300, bbox_inches='tight')
        fig_legend.savefig(os.path.join('/scratch/gobi1/sana/TSX_results',data, 'legend_%d_%s.pdf' %(subject, self.generator_type)), dpi=300, bbox_inches='tight')
        return max_imp_FCC, importance, max_imp_occ, importance_occ, max_imp_occ_aug, importance_occ_aug, max_imp_sen, sensitivity_analysis_importance

    def final_reported_plots(self, samples_to_analyze):
        testset = list(self.test_loader.dataset)
        baseline_methods = ['FFC', 'AFO', 'FO', 'SA']#, 'lime']

        for subject in samples_to_analyze:
            f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
            for method_ind, method in enumerate(baseline_methods):
                signals, label_o = testset[subject]
                if self.data =='mimic':
                    label_o = torch.zeros(signals.shape[-1])
                    for t in range(1, signals.shape[-1]):
                        self.risk_predictor.eval()
                        label_o[t-1] = self.risk_predictor(signals[:, 0:t].unsqueeze(0)).item()
                with open('./examples/%s/baseline_importance_sample_%d.json'%(self.data, subject)) as config_file:
                    important_observations = json.load(config_file)[method]
                ax = [ax2, ax3, ax4, ax5][method_ind]
                ax.plot(label_o[1:].cpu().numpy(), linewidth=3, color='r',  label='Model output')
                for observation in important_observations:
                    ref_ind = observation[0]
                    imp_t = observation[1]
                    c = color_map[ref_ind]
                    ax.plot(np.array(signals[ref_ind, 1:]), linewidth=3, color=c,  label='%s' % (self.feature_map[ref_ind]))
                    ax.axvspan(imp_t-1, imp_t+1, facecolor=c, alpha=0.3, hatch='/', label='%s score: %.3f' % (self.feature_map[ref_ind], observation[2]))
                    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=10, fontsize=18)
                    ax.set_title('%s'%method, fontweight='bold', fontsize=34)
                    ax.set_ylabel('signal value', fontweight='bold', fontsize=24)
                    ax.tick_params(axis='both', labelsize=26)
            from TSX.generator_baselines import find_true_gen_importance
            true_importance = find_true_gen_importance(signals.to(self.device), self.data)
            for feature in range(signals.shape[0]):
                imp_t = np.argmax(true_importance[feature,:])
                c = color_map[feature]
                ax1.plot(np.array(signals[feature, 1:]), linewidth=3, color=c, label='%s' % (self.feature_map[feature]))
                ax1.axvspan(imp_t - 1, imp_t + 1, facecolor=c, alpha=0.3, hatch='/',
                           label='%s score: %.3f' % (self.feature_map[feature], true_importance[feature,imp_t]))
                # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=10, fontsize=18)
                ax1.set_title('Ground Truth Generator', fontweight='bold', fontsize=34)
                ax1.set_ylabel('signal value', fontweight='bold', fontsize=24)
                ax1.tick_params(axis='both', labelsize=26)
            f.set_figheight(25)
            f.set_figwidth(30)
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(os.path.join('./examples', self.data, 'baselines_%d_%s.pdf' % (subject, self.generator_type)), dpi=300,
                        orientation='landscape',
                        bbox_inches='tight')

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

            # ckpt_path='./ckpt_pathckpt/' + data + '/'
            for i, sig_ind in enumerate(range(0,self.feature_size)):
                print('loading feature from:', self.ckpt_path)
                if not self.generator_type=='carry_forward_generator':
                    if 'joint' in self.generator_type:
                        self.generator.load_state_dict(torch.load(os.path.join(self.ckpt_path, '%s.pt'%self.generator_type)))
                    else:
                        if self.historical:
                            self.generator.load_state_dict(torch.load(os.path.join(self.ckpt_path,'%d_%s.pt'%(sig_ind, self.generator_type))))
                        else:
                            self.generator.load_state_dict(torch.load(os.path.join(self.ckpt_path,'%d_%s_nohist.pt'%(sig_ind, self.generator_type))))

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
                #label_scaled = self.patient_data.scaler_y.inverse_transform(np.reshape(np.array(label),[-1,1]))
                #label_scaled = np.reshape(label_scaled,[1,-1])
                label_scaled=label

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
        train_start_time = time.time()
        if 'joint' in self.generator_type:
            train_joint_feature_generator(self.generator, self.train_loader, self.valid_loader, generator_type=self.generator_type , n_epochs=n_epochs)
        else:
            for feature_to_predict in range(0,n_features):
                print('**** training to sample feature: ', feature_to_predict)
                self.generator = FeatureGenerator(self.feature_size, self.historical, hidden_size=self.generator_hidden_size, prediction_size=self.prediction_size,data=self.data,conditional=self.conditional).to(self.device)
                train_feature_generator(self.generator, self.train_loader, self.valid_loader, self.generator_type, feature_to_predict, n_epochs=n_epochs, historical=self.historical)
        print("Training time for %s = "%(self.generator_type), time.time()-train_start_time)


    def _get_feature_importance_FFC(self, signal, sig_ind, n_samples=10, learned_risk=True, tvec=None):
        self.generator.eval()
        # signal = signal.to(self.device)
        risks = []
        importance = []
        if tvec is None:
            tvec = range(1,signal.shape[1])

        for t in tvec:
            if self.simulation:
                if not learned_risk:
                    risk = self.risk_predictor(signal.cpu().detach().numpy(), t)
                else:
                    risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t+1)).item()
            else:
                risk = self.risk_predictor(signal[:,0:t+self.generator.prediction_size].view(1, signal.shape[0], t+self.generator.prediction_size)).item()

            generator_predicted_risks = []
            conditional_predicted_risks = []

            # dist_mean, dist_covariance = self.generator.likelihood_distribution(signal[:, :t].unsqueeze(0))
            # dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=dist_mean, covariance_matrix=dist_covariance)
            # mean_cond, covariance_cond = test_cond(dist_mean, dist_covariance, sig_ind, signal[sig_ind,t].unsqueeze(0).to(self.device))
            # dist_cond = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean_cond[:,:,0],
            #                                                                        covariance_matrix=covariance_cond)
            for _ in range(n_samples):
                # Replace signal with random sample from the distribution if feature_occlusion==True,
                # else use the generator model to estimate the value
                # x_hat_t = self.generator.forward_joint(signal[:, :t].unsqueeze(0))

                # Definition I
                x_hat_t_cond, _ = self.generator.forward(signal[:,t], signal[:, :t].unsqueeze(0), sig_ind, method='m1')
                # Definition II
                # x_hat_t_cond, _ = self.generator.forward(signal[:,t], signal[:, :t].unsqueeze(0), sig_ind, method='c1')

                # x_hat_t = dist.rsample()
                # sample_cond = dist_cond.rsample()
                # x_hat_t_cond = torch.cat([sample_cond[0, 0:sig_ind], signal[sig_ind,t].unsqueeze(-1).to(self.device), sample_cond[0, sig_ind:]])

                # predicted_signal = signal[:,0:t+1].clone()
                # predicted_signal[:,-1] = x_hat_t
                predicted_signal_conditional = signal[:,0:t+1].clone()
                predicted_signal_conditional[:, -1] = x_hat_t_cond
                # print(x_hat_t_cond, signal[:,t])

                if self.simulation and not learned_risk:
                    # predicted_risk = self.risk_predictor(predicted_signal.cpu().detach().numpy(), t)
                    conditional_predicted_risk = self.risk_predictor(predicted_signal_conditional.cpu().detach().numpy(), t)
                else:
                    # predicted_risk = self.risk_predictor(predicted_signal.unsqueeze(0).to(self.device)).item()
                    conditional_predicted_risk = self.risk_predictor(predicted_signal_conditional.unsqueeze(0).to(self.device)).item()
                # generator_predicted_risks.append(predicted_risk)
                conditional_predicted_risks.append(conditional_predicted_risk)

            # Definition I
            conditional_predicted_risks = np.array(conditional_predicted_risks)
            entropy = -1 * risk * np.log2(risk) - (1.-risk)*np.log2(1.-risk)
            conditional_entropy = -1 * np.multiply(conditional_predicted_risks, np.log2(conditional_predicted_risks)) - np.multiply((1.-conditional_predicted_risks), np.log2((1.-conditional_predicted_risks)))
            imp = max(0,np.mean(conditional_entropy - entropy))

            # Definition II
            # generator_predicted_risks = np.array(generator_predicted_risks)
            # conditional_predicted_risks = np.array(conditional_predicted_risks)
            # entropy = -1*(np.mean(generator_predicted_risks)*np.log2(np.mean(generator_predicted_risks))) - (np.mean(1.-generator_predicted_risks) * np.log2(np.mean(1.-generator_predicted_risks)))
            # conditional_entropy = -1*(np.mean(conditional_predicted_risks)*np.log2(np.mean(conditional_predicted_risks))) - (np.mean(1.-conditional_predicted_risks) * np.log2(np.mean(1.-conditional_predicted_risks)))
            # imp = entropy - conditional_entropy

            # entropy = np.mean(-1*(np.multiply(generator_predicted_risks, np.log2(generator_predicted_risks))) - (np.multiply((1.-generator_predicted_risks), np.log2(1.-generator_predicted_risks))))
            # conditional_entropy = np.mean(-1*(np.multiply(conditional_predicted_risks, np.log2(conditional_predicted_risks))) - (np.multiply((1-conditional_predicted_risks), np.log2(1-conditional_predicted_risks))))
            importance.append(imp)

            risks.append(risk)
        return risks, importance


    def _get_feature_importance(self, signal, sig_ind, n_samples=10, mode="feature_occlusion", learned_risk=True, tvec=None, method='m1'):
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
                    risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t+1)).item()
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
                    prediction = torch.Tensor(np.array([np.random.uniform(-3,3)]).reshape(-1)).to(self.device)
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
                    predicted_signal_t, _ = self.generator(signal[:, t], signal[:, 0:t].view(1, signal.size(0), t), sig_ind, method)
                    if mode=="combined":
                        if self.risk_predictor(signal[:,0:t].view(1, signal.shape[0], t)).item() > 0.5:
                            prediction = torch.Tensor(self._find_closest(feature_dist_0, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)
                        else:
                            prediction = torch.Tensor(self._find_closest(feature_dist_1, prediction.cpu().detach().numpy()).reshape(-1)).to(self.device)
                    predicted_signal = signal[:,0:t+self.generator.prediction_size].clone()
                    predicted_signal[:,t:t+self.generator.prediction_size] = predicted_signal_t.view(-1, 1)

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

            diff_imp = abs(predicted_risks-risk)
            mean_imp = np.mean(predicted_risks,0)
            std_imp = np.std(predicted_risks, 0)
            mean_predicted_risk.append(mean_imp)
            std_predicted_risk.append(std_imp)
            if (method=='c1' or method=='inf') and (mode=="generator" or mode=="combined"):
                importance.append(1. - 1./(1+np.exp(-10*(abs(mean_imp - risk)-0.5)))  )
            else:
                def assign_importance_IG(cf_prediction, prediction):
                    cf_entropy = [-1*(i*log(i) + (1-i)*log(1-i)) for i in cf_prediction]
                    entropy = -1*(prediction*log(prediction) + (1-prediction)*log(1-prediction))
                    IG = np.array(cf_entropy) - entropy
                    return np.mean(IG)
                #importance.append(abs(mean_imp-risk))
                importance.append(np.mean(diff_imp))
                #importance.append(assign_importance_IG(predicted_risks, risk))
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
                for s in range(len(ind_list)):
                    out = self.risk_predictor(signal_t)
                    out[s].backward()#retain_graph=True)
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
                max_imp_afo = []
                importance = np.zeros((len(signals_analyze),start_point))
                importance_occ = np.zeros((len(signals_analyze), start_point))
                importance_afo = np.zeros((len(signals_analyze), start_point))

                self.risk_predictor.eval()
                prediction = int(self.risk_predictor(signal[:,:start_point+1].view(1,signal.shape[0],start_point+1)).item()>0.5)
                if prediction<0.7 or prediction!=label:
                    continue
                for i in range(27):
                    label, importance[i, :], _, _ = self._get_feature_importance(signal[:,:start_point+1], sig_ind=i, n_samples=10,
                                                                                mode='generator', learned_risk=self.learned_risk)
                    _, importance_occ[i, :], _, _ = self._get_feature_importance(signal[:,:start_point+1], sig_ind=i, n_samples=10,
                                                                                 mode="feature_occlusion", learned_risk=self.learned_risk)
                    _, importance_afo[i, :], _, _ = self._get_feature_importance(signal[:,:start_point+1], sig_ind=i, n_samples=10,
                                                                                 mode="augmented_feature_occlusion", learned_risk=self.learned_risk)
                    max_imp_FCC.append((i, max(importance[i, :])))
                    max_imp_occ.append((i, max(importance_occ[i, :])))
                    max_imp_afo.append((i, max(importance_afo[i, :])))
                    max_imp_sen.append((i, sensitivity_analysis[ind,i,start_point+1]))
                max_imp_FCC.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_occ.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_sen.sort(key=lambda pair: pair[1], reverse=True)
                max_imp_afo.sort(key=lambda pair: pair[1], reverse=True)
                print(max_imp_afo)

                df.loc[-1] = [subject,intervention_ID,'FCC',max_imp_FCC[0][0],max_imp_FCC[1][0],max_imp_FCC[2][0]]  # adding a row
                df.index = df.index + 1
                df.loc[-1] = [subject,intervention_ID,'f_occ',max_imp_occ[0][0],max_imp_occ[1][0],max_imp_occ[2][0]]  # adding a row
                df.index = df.index + 1
                df.loc[-1] = [subject,intervention_ID,'afo',max_imp_afo[0][0],max_imp_afo[1][0],max_imp_afo[2][0]]  # adding a row
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
        afo_df = df.loc[df['method'] == 'afo']
        occ_df = df.loc[df['method'] == 'f_occ']
        sen_df = df.loc[df['method'] == 'sensitivity']
        fcc_dist = np.sort(np.array(fcc_df[['top1','top2','top3']]).reshape(-1,))
        afo_dist = np.sort(np.array(afo_df[['top1', 'top2', 'top3']]).reshape(-1, ))
        occ_dist = np.sort(np.array(occ_df[['top1', 'top2', 'top3']]).reshape(-1, ))
        sen_dist = np.sort(np.array(sen_df[['top1', 'top2', 'top3']]).reshape(-1, ))


        fcc_top = self._create_pairs(self._find_count(fcc_dist))[0:4]
        afo_top = self._create_pairs(self._find_count(afo_dist))[0:4]
        occ_top = self._create_pairs(self._find_count(occ_dist))[0:4]
        sen_top = self._create_pairs(self._find_count(sen_dist))[0:4]
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharey=True)
        ax1.bar([self.feature_map[x[0]] for x in fcc_top], [x[1] for x in fcc_top], color=[color_map[x[0]] for x in fcc_top])
        ax2.bar([self.feature_map[x[0]] for x in afo_top], [x[1] for x in afo_top], color=[color_map[x[0]] for x in afo_top])
        ax3.bar([self.feature_map[x[0]] for x in occ_top], [x[1] for x in occ_top], color=[color_map[x[0]] for x in occ_top])
        ax4.bar([self.feature_map[x[0]] for x in sen_top], [x[1] for x in sen_top], color=[color_map[x[0]] for x in sen_top])
        f.suptitle('%s'%(intervention_list[intervention_ID]), fontweight='bold', fontsize=36)
        ax1.set_title('FFC', fontsize=32, fontweight='bold')
        ax2.set_title('AFO', fontsize=32, fontweight='bold')
        ax3.set_title('FO', fontsize=32, fontweight='bold')
        ax4.set_title('Sensitivity analysis', fontsize=32, fontweight='bold')
        ax1.tick_params(labelsize=30)
        ax2.tick_params(labelsize=30)
        ax3.tick_params(labelsize=30)
        ax4.tick_params(labelsize=30)
        plt.subplots_adjust(hspace=0.6)
        f.set_figheight(12)
        f.set_figwidth(15)
        if not os.path.exists('./plots/distributions'):
            os.mkdir('./plots/distributions')
        plt.savefig('./plots/distributions/top_%s.pdf'%(intervention_list[intervention_ID]), dpi=300, bbox_inches='tight')

        f, (ax1,ax2,ax3) = plt.subplots(3, sharex=True)
        ax1.bar(self.feature_map, self._find_count(fcc_dist))
        ax2.bar(self.feature_map, self._find_count(occ_dist))
        ax3.bar(self.feature_map, self._find_count(sen_dist))
        ax1.set_title('FFC importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        ax2.set_title('feature occlusion importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        ax3.set_title('sensitivity analysis importance distribution for %s'%(intervention_list[intervention_ID]), fontsize=20)
        f.set_figheight(20)
        f.set_figwidth(15)
        plt.savefig('./plots/distributions/%s.pdf'%(intervention_list[intervention_ID]))

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


# class MethodComparison(Experiment):
#     """ Baseline explainability methods
#     """
#     def __init__(self,  train_loader, valid_loader, test_loader, feature_size, data_class, experiment='method_comparison', data='mimic', baseline_methods=['FFC', 'AFO', 'lime', 'FO', 'sensitivity']):
#         super(MethodComparison, self).__init__(train_loader, valid_loader, test_loader, data=data)
#         self.experiment = experiment
#         self.data_class = data_class
#         self.ckpt_path = os.path.join('./ckpt/', self.data)
#         self.input_size = feature_size
#         self.baseline_methods = baseline_methods
#
#     def run(self, train, n_epochs=60, samples_to_analyze=MIMIC_TEST_SAMPLES):
#         self.experiment_map = {'FFC':FeatureGeneratorExplainer, 'AFO':FeatureGeneratorExplainer, 'lime':BaselineExplainer, 'FO':FeatureGeneratorExplainer, 'sensitivity':FeatureGeneratorExplainer}
#         if data == 'mimic':
#             self.timeseries_feature_size = feature_size - 4
#         else:
#             self.timeseries_feature_size = feature_size
#
#         # Build the risk predictor and load checkpoint
#         with open('config.json') as config_file:
#             configs = json.load(config_file)[data]['risk_predictor']
#         if self.data == 'simulation':
#             if not self.learned_risk:
#                 self.risk_predictor = lambda signal,t:logistic(2.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t] - 1))
#             else:
#                 self.risk_predictor = EncoderRNN(configs['feature_size'], hidden_size=configs['encoding_size'],
#                                                  rnn=configs['rnn_type'], regres=True, return_all=False, data=data)
#             self.feature_map = feature_map_simulation
#         else:
#             if self.data == 'mimic':
#                 self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
#                                                  rnn=configs['rnn_type'], regres=True, data=data)
#                 self.feature_map = feature_map_mimic
#                 self.risk_predictor = self.risk_predictor.to(self.device)
#             elif self.data == 'ghg':
#                 self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
#                                                  rnn=configs['rnn_type'], regres=True, data=data)
#                 self.feature_map = feature_map_ghg
#                 self.risk_predictor = self.risk_predictor.to(self.device)
#         self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor.pt')))
#         self.risk_predictor.to(self.device)
#         self.risk_predictor.eval()
#
#         self.train(n_epochs=n_epochs, learn_rt=self.data=='ghg')
#         testset = list(self.test_loader.dataset)
#         test_signals = torch.stack(([x[0] for x in testset])).to(self.device)
#         matrix_test_dataset = test_signals.mean(dim=2).cpu().numpy()
#         for test_sample in samples_to_analyze:
#             exp = self.explainer.explain_instance(matrix_test_dataset[test_sample], self.predictor_wrapper, num_features=4, top_labels=2)
#             print("Most important features for sample %d: "%(test_sample), exp.as_list())
#
#     def train(self, n_epochs, learn_rt=False):
#         trainset = list(self.train_loader.dataset)
#         signals = torch.stack(([x[0] for x in trainset])).to(self.device)
#         matrix_train_dataset = signals.mean(dim=2).cpu().numpy()
#         if self.baseline_method == 'lime':
#             self.explainer = lime.lime_tabular.LimeTabularExplainer(matrix_train_dataset, feature_names=self.feature_map+['gender', 'age', 'ethnicity', 'first_icu_stay'], discretize_continuous=True)
