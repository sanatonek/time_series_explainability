from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.models import DeepKnn
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer, BaselineExplainer
import torch
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle as pkl
import sys
import glob
import re
sys.path.append(os.path.join(os.path.dirname(__file__),".."))


feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2','Glucose', 'Temp', 'gender','age','ethnicity','first_icu_stay']

def parse_lime_results(arr,Tt,n_features,data='ghg'):
    if data=='mimic':
        tvec = range(Tt)
    else:
        tvec = range(1,Tt+1,50)
    lime_res = np.zeros((n_features,len(tvec)))
    for i,t in enumerate(tvec):
        impt_time_t = arr['lime']['imp'][0][t]
        for ff in impt_time_t:
            parse_str = re.split('> | < | <= | >= | =',ff[0])
            #print(parse_str)
            if len(parse_str)==2: #feature number is always first
                feature_idx = parse_str[0].strip()
                if data=='ghg':
                    feature_idx = int(feature_idx)-1
                else:
                    #print(np.where(np.array(feature_map_mimic)==feature_idx))
                    feature_idx = np.where(np.array(feature_map_mimic)==feature_idx)[0][0]
            elif len(parse_str)==3: #feature number is always in the middle
                feature_idx = parse_str[1].strip()
                if data=='ghg':
                    feature_idx = int(feature_idx)-1
                else:
                    #print(np.where(np.array(feature_map_mimic)==feature_idx))
                    feature_idx = np.where(np.array(feature_map_mimic)==str(feature_idx))[0][0]
            #print(feature_idx)
            feature_val = abs(arr['lime']['imp'][0][t][0][1])
            #if feature_idx<n_features:
            lime_res[int(feature_idx),i]=feature_val
    return lime_res

def main(experiment, train, user, data,n_features_to_use=3):
    #sys.stdout = open('/scratch/gobi1/shalmali/global_importance_'+data+'.txt', 'w')
    filelist = glob.glob(os.path.join('/scratch/gobi1/%s/TSX_results'%user,data,'results_*.pkl'))
    
    N=len(filelist)
    with open(filelist[0],'rb') as f:
        arr = pkl.load(f)
    
    n_features=arr['FFC']['imp'].shape[0]
    Tt=arr['FFC']['imp'].shape[1]

    y_ffc=np.zeros((N,n_features))
    y_afo=np.zeros((N,n_features))
    y_suresh=np.zeros((N,n_features))
    y_sens=np.zeros((N,n_features))
    y_lime=np.zeros((N,n_features))

    for n,file in enumerate(filelist):
        with open(file,'rb') as f:
            arr = pkl.load(f)
        
        y_ffc[n,:] = arr['FFC']['imp'].sum(1)
        y_afo[n,:] = arr['AFO']['imp'].sum(1)
        y_suresh[n,:] = arr['Suresh_et_al']['imp'].sum(1)
        y_sens[n,:] = arr['Sens']['imp'][:len(arr['FFC']['imp']),1:].sum(1)
        y_lime[n,:] = parse_lime_results(arr,Tt,n_features,data=data).sum(1)
    
    y_rank_ffc = np.flip(np.argsort(y_ffc.sum(0)).flatten())# sorted in order of relevance
    y_rank_afo = np.flip(np.argsort(y_afo.sum(0)).flatten())# sorted in order of relevance
    y_rank_suresh = np.flip(np.argsort(y_suresh.sum(0)).flatten())# sorted in order of relevance
    y_rank_sens = np.flip(np.argsort(y_sens.sum(0)).flatten())# sorted in order of relevance
    y_rank_lime = np.flip(np.argsort(y_lime.sum(0)).flatten())# sorted in order of relevance
    ranked_features = {'ffc': y_rank_ffc, 'afo':y_rank_afo,'suresh':y_rank_suresh,'sens':y_rank_sens,'lime':y_rank_lime}
    
    with open('config.json') as config_file:
        configs = json.load(config_file)[data][experiment]

    methods = ranked_features.keys()

    for m in methods:
        print('Experiment with 5 most relevant features: ', m)
        feature_rank = ranked_features[m]
       
        for ff in [n_features_to_use]:
            features = feature_rank[:ff]
            print('using features',features)

            if data == 'mimic':
                p_data, train_loader, valid_loader, test_loader = load_data(batch_size=configs['batch_size'],
                                                                    path='./data',features=features)
                feature_size = p_data.feature_size
            elif data == 'ghg':
                p_data, train_loader, valid_loader, test_loader = load_ghg_data(configs['batch_size'],features=features)
                feature_size = p_data.feature_size
                print(feature_size)
            elif data == 'simulation_spike':
                p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data_generator/data/simulated_data',data_type='spike',features=features)
                feature_size = p_data.shape[1]

            elif data == 'simulation':
                p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data/simulated_data',features=features)
                feature_size = p_data.shape[1]

            if data=='simulation_spike':
                data='simulation'
                spike_data=True
            else:
                spike_data=False


            print('training on ', feature_size, ' features!')
            
            exp = EncoderPredictor(train_loader, valid_loader, test_loader, feature_size, configs['encoding_size'], rnn_type=configs['rnn_type'], data=data)
            exp.run(train=train, n_epochs=configs['n_epochs'])   
            
    n_features_to_remove = 10 #add/remove same number for now
    #Exp 1 remove and evaluate
    for m in methods:
        print('Experiment for removing features using method: ', m)
        feature_rank = ranked_features[m]
        
        #for ff in range(min(n_features-1,n_features_to_remove)):
        for ff in [n_features_to_remove]:
            features = [ elem for elem in list(range(n_features)) if elem not in feature_rank[:ff]]
            #print('using features:', features)

            if data == 'mimic':
                p_data, train_loader, valid_loader, test_loader = load_data(batch_size=configs['batch_size'],
                                                                    path='./data',features=features)
                feature_size = p_data.feature_size
            elif data == 'ghg':
                p_data, train_loader, valid_loader, test_loader = load_ghg_data(configs['batch_size'],features=features)
                feature_size = p_data.feature_size
                print(feature_size)
            elif data == 'simulation_spike':
                p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data_generator/data/simulated_data',data_type='spike',features=features)
                feature_size = p_data.shape[1]

            elif data == 'simulation':
                p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data/simulated_data',features=features)
                feature_size = p_data.shape[1]

            if data=='simulation_spike':
                data='simulation'
                spike_data=True
            else:
                spike_data=False


            print('training on ', feature_size, ' features!')
            
            exp = EncoderPredictor(train_loader, valid_loader, test_loader, feature_size, configs['encoding_size'], rnn_type=configs['rnn_type'], data=data)
            exp.run(train=train, n_epochs=configs['n_epochs'])
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--model', type=str, default='feature_generator_explainer', help='Prediction model')
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--user', type=str, default='sana')
    parser.add_argument('--n_features', type=int, default=5)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, data=args.data, user=args.user)
