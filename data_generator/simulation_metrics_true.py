#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import pickle as pkl

def parse_lime_results(arr,Tt,n_features):
    lime_res = np.zeros((n_features,Tt))
    for t in range(Tt):
        parse_str = np.array(arr['lime']['imp'][0][t][0][0].split(' '))
        feature_idx = np.where(np.array(parse_str)=='feature')[0][0]+1
        feature_val = abs(arr['lime']['imp'][0][t][0][1])
        lime_res[int(parse_str[feature_idx])-1,t]=feature_val
    return lime_res


#preprocess before metric collection
data='simulation'
filelist = glob.glob('/scratch/gobi1/shalmali/TSX_results/'+data+'/results_*cv_0.pkl')

N=len(filelist)
with open(filelist[0],'rb') as f:
    arr = pkl.load(f)
#print(arr.keys())
n_features,Tt = arr['FFC']['imp'].shape

y_true=np.zeros(n_features*N*Tt)
y_ffc=np.zeros(n_features*N*Tt)
y_cond=np.zeros(n_features*N*Tt)
y_afo=np.zeros(n_features*N*Tt)
y_suresh=np.zeros(n_features*N*Tt)
y_sens=np.zeros(n_features*N*Tt)
y_lime=np.zeros(n_features*N*Tt)

y_binary_ffc=np.zeros(n_features*N*Tt)
y_binary_cond=np.zeros(n_features*N*Tt)
y_binary_afo=np.zeros(n_features*N*Tt)
y_binary_suresh=np.zeros(n_features*N*Tt)
y_binary_sens=np.zeros(n_features*N*Tt)
y_binary_lime=np.zeros(n_features*N*Tt)

y_true_rk0=np.zeros((n_features, N*Tt)).T
y_true_rk=np.zeros((n_features, N*Tt)).T
y_ffc_rk=np.zeros((n_features, N*Tt)).T
y_afo_rk=np.zeros((n_features, N*Tt)).T
y_suresh_rk=np.zeros((n_features, N*Tt)).T
y_sens_rk=np.zeros((n_features, N*Tt)).T

plot=0

nn=N
thrs = np.linspace(0,0.5,50)

tpr_ffc=[]
fpr_ffc=[]
precision_ffc=[]
recall_ffc=[]

tpr_afo=[]
fpr_afo=[]
precision_afo=[]
recall_afo=[]

tpr_suresh=[]
fpr_suresh=[]
precision_suresh=[]
recall_suresh=[]

tpr_sens=[]
fpr_sens=[]
precision_sens=[]
recall_sens=[]

tpr_lime=[]
fpr_lime=[]
precision_lime=[]
recall_lime=[]

for th in thrs:
    for n,file in enumerate(filelist):
        with open(file,'rb') as f:
            arr = pkl.load(f)

        n_obs=Tt*n_features
        
        if data!='simulation_spike':
            y_true[n*n_obs:(n+1)*n_obs] = arr['gt'][:,1:].flatten()
        else:
            gt_array = np.zeros((n_features,Tt))
            gt_array[0,:] = arr['gt'][1:]
            y_true[n*n_obs:(n+1)*n_obs] = gt_array.flatten()

        y_ffc[n*n_obs:(n+1)*n_obs] = arr['FFC']['imp'].flatten()

        
        
        y_binary_ffc[n*n_obs:(n+1)*n_obs] = arr['FFC']['imp'].flatten()
        y_binary_ffc[y_binary_ffc>=th] = 1
        y_binary_ffc[y_binary_ffc<th] = 1
        
       

     
    #print metrics
    auc_ffc= metrics.roc_auc_score(y_true, y_ffc)


    # recall/sensitivity
    report_ffc = metrics.classification_report(y_true, y_binary_ffc,output_dict=True)['1.0']

        
    # auprc
    auprc_ffc= metrics.average_precision_score(y_true, y_ffc)
 

    '''
    print('FFC - AUC: ',auc_ffc, ' Sensitivity: ',report_ffc['recall'], ' AUPRC: ',  auprc_ffc)
    print('AFO - AUC: ',auc_afo, ' Sensitivity: ',report_afo['recall'], ' AUPRC: ',  auprc_afo)
    print('Suresh - AUC: ',auc_suresh, ' Sensitivity: ',report_suresh['recall'], ' AUPRC: ',  auprc_suresh)
    print('Sens - AUC: ',auc_sens, ' Sensitivity: ',report_sens['recall'], ' AUPRC: ',  auprc_sens)
    '''

    print('---------------------------------------------------thr:', th)
    print('FFC & ', round(auc_ffc,4),  ' & ',  round(auprc_ffc,4) ,'&' , report_ffc['precision'],'&',report_ffc['recall'], '&' ,report_ffc['f1-score'],'\\\\')

    precision_ffc.append(report_ffc['precision'])
    recall_ffc.append(report_ffc['recall'])

print('aupr ffc - final :', metrics.auc(precision_ffc,recall_ffc,thrs))


