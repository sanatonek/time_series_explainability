#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import pickle as pkl
import glob
import os,sys

def parse_lime_results(arr,Tt,n_features):
    lime_res = np.zeros((n_features,Tt))
    for t in range(Tt):
        parse_str = np.array(arr['lime']['imp'][0][t][0][0].split(' '))
        feature_idx = np.where(np.array(parse_str)=='feature')[0][0]+1
        feature_val = abs(arr['lime']['imp'][0][t][0][1])
        lime_res[int(parse_str[feature_idx])-1,t]=feature_val
    return lime_res


#preprocess before metric collection
data='simulation_spike'
fpath='/scratch/gobi1/shalmali/TSX_results/'
#predictor_model='attention'
predictor_model='RNN'
filelist = glob.glob(os.path.join(fpath,data,predictor_model,'results_*cv_0.pkl'))
#print(filelist)
#filelist = glob.glob('/scratch/gobi1/sana/TSX_results/simulation_non_stationary/results_*.pkl')

N=len(filelist)
with open(filelist[0],'rb') as f:
    arr = pkl.load(f)
#print(arr.keys())

n_features,Tt = arr['FFC']['imp'].shape
plot=0

thrs = np.linspace(0,1,10)

tpr_ffc=[]
fpr_ffc=[]
precision_ffc=[]
recall_ffc=[]
auc_ffc=[]
auprc_ffc=[]

tpr_cond=[]
fpr_cond=[]
precision_cond=[]
recall_cond=[]
auc_cond=[]
auprc_cond=[]

tpr_att=[]
fpr_att=[]
precision_att=[]
recall_att=[]
auc_att=[]
auprc_att=[]

tpr_afo=[]
fpr_afo=[]
precision_afo=[]
recall_afo=[]
auc_afo=[]
auprc_afo=[]

tpr_suresh=[]
fpr_suresh=[]
precision_suresh=[]
recall_suresh=[]
auc_fo=[]
auprc_fo=[]

tpr_sens=[]
fpr_sens=[]
precision_sens=[]
recall_sens=[]
auc_sens=[]
auprc_sens=[]

tpr_lime=[]
fpr_lime=[]
precision_lime=[]
recall_lime=[]
auc_lime=[]
auprc_lime=[]

tpr_true_gen=[]
fpr_true_gen=[]
precision_true_gen=[]
recall_true_gen=[]
auc_true_gen=[]
auprc_true_gen=[]
th=0.5

for cv in [0,2]:
    #for th in thrs:
    #for n,file in enumerate(filelist):
    filelist=[]
    for n in range(0,100):
        filelist.append(glob.glob(os.path.join(fpath,data,predictor_model,'results_'+str(n)+'cv_'+str(cv)+'.pkl'))[0])
    N=len(filelist)

    y_true=np.zeros(n_features*N*Tt)
    y_ffc=np.zeros(n_features*N*Tt)
    y_att=np.zeros(n_features*N*Tt)
    y_cond=np.zeros(n_features*N*Tt)
    y_afo=np.zeros(n_features*N*Tt)
    y_suresh=np.zeros(n_features*N*Tt)
    y_sens=np.zeros(n_features*N*Tt)
    y_lime=np.zeros(n_features*N*Tt)
    y_true_gen=np.zeros(n_features*N*Tt)

    y_binary_ffc=np.zeros(n_features*N*Tt)
    y_binary_att=np.zeros(n_features*N*Tt)
    y_binary_afo=np.zeros(n_features*N*Tt)
    y_binary_cond=np.zeros(n_features*N*Tt)
    y_binary_suresh=np.zeros(n_features*N*Tt)
    y_binary_sens=np.zeros(n_features*N*Tt)
    y_binary_lime=np.zeros(n_features*N*Tt)
    y_binary_true_gen=np.zeros(n_features*N*Tt)

    y_true_rk0=np.zeros((n_features, N*Tt)).T
    y_true_rk=np.zeros((n_features, N*Tt)).T
    y_ffc_rk=np.zeros((n_features, N*Tt)).T
    y_afo_rk=np.zeros((n_features, N*Tt)).T
    y_suresh_rk=np.zeros((n_features, N*Tt)).T
    y_sens_rk=np.zeros((n_features, N*Tt)).T


    #for nn, n in enumerate(list(range(0,40))):
    for nn,fname in enumerate(filelist):
        #if cv==0:
        # file = glob.glob(os.path.join(fpath,data,'results_'+ str(n) + '.pkl'))[0]
        #file = glob.glob(os.path.join('/scratch/gobi1/sana/TSX_results', data, 'results_*cv_0.pkl'))[0]
        #print(fname)
        #else:
        #print(fpath, data)
        #print(os.path.join(fpath,data,'/results_'+str(n)+ 'cv_'+str(cv)+'.pkl'))
        #fname = glob.glob(os.path.join(fpath,data,predictor_model,'results_'+str(n)+ 'cv_'+str(cv)+'.pkl'))[0]
        
        with open(fname,'rb') as f:
            arr = pkl.load(f)
           
        if len(arr.keys())<7:
            continue
        #print(file)
        n_obs=Tt*n_features

        #if data!='simulation_spike':
            #print(arr.keys())
        #print(y_true)
        y_true[nn*n_obs:(nn+1)*n_obs] = arr['gt'][:,1:].flatten()
        #else:
        #    gt_array = np.zeros((n_features,Tt))
        #    gt_array[0,:] = arr['gt'][1:]
        #    y_true[nn*n_obs:(nn+1)*n_obs] = gt_array.flatten()

        y_ffc[nn*n_obs:(nn+1)*n_obs] = np.clip(arr['FFC']['imp'].flatten(),0,10**8)
        #print(y_ffc)
        #y_ffc[nn * n_obs:(nn + 1) * n_obs] = 1./(1.+np.exp(1000*y_ffc[nn*n_obs:(nn+1)*n_obs]))
        #y_ffc[nn * n_obs:(nn + 1) * n_obs] = 1./(1.+np.exp(0.00000005*y_ffc[nn*n_obs:(nn+1)*n_obs]))
        #print(np.where(np.isnan(y_ffc)),np.where(np.isinf(y_ffc)))
        y_afo[nn*n_obs:(nn+1)*n_obs] = arr['AFO']['imp'].flatten()
        y_att[nn*n_obs:(nn+1)*n_obs] = arr['attention']['imp'].flatten()
        y_cond[nn*n_obs:(nn+1)*n_obs] = arr['conditional']['imp'].flatten()
        y_suresh[nn*n_obs:(nn+1)*n_obs] = arr['Suresh_et_al']['imp'].flatten()
        y_sens[nn*n_obs:(nn+1)*n_obs] = arr['Sens']['imp'][:,1:].flatten()
        y_lime[nn*n_obs:(nn+1)*n_obs] = parse_lime_results(arr,Tt,n_features).flatten()

        #file = glob.glob('/scratch/gobi1/shalmali/'+data+'/results_true_'+str(n)+'.pkl')[0]
        #with open(file,'rb') as f:
        #    arr_true = pkl.load(f)
        #y_true_gen[nn*n_obs:(nn+1)*n_obs] = arr_true.flatten()


    #print metrics
    auc_ffc_cv= metrics.roc_auc_score(y_true, y_ffc)
    auc_att_cv= metrics.roc_auc_score(y_true, y_att)
    auc_cond_cv= metrics.roc_auc_score(y_true, y_cond)
    auc_afo_cv= metrics.roc_auc_score(y_true, y_afo)
    auc_suresh_cv= metrics.roc_auc_score(y_true, y_suresh)
    auc_sens_cv= metrics.roc_auc_score(y_true, y_sens)
    auc_lime_cv= metrics.roc_auc_score(y_true, y_lime)
    auc_true_gen_cv= metrics.roc_auc_score(y_true, y_true_gen)


    # auprc
    auprc_ffc_cv= metrics.average_precision_score(y_true, y_ffc)
    auprc_afo_cv= metrics.average_precision_score(y_true, y_afo)
    auprc_cond_cv= metrics.average_precision_score(y_true, y_cond)
    auprc_att_cv= metrics.average_precision_score(y_true, y_att)
    auprc_suresh_cv= metrics.average_precision_score(y_true, y_suresh)
    auprc_sens_cv= metrics.average_precision_score(y_true, y_sens)
    auprc_lime_cv= metrics.average_precision_score(y_true, y_lime)
    auprc_true_gen_cv= metrics.average_precision_score(y_true, y_true_gen)

    '''
    print('FFC - AUC: ',auc_ffc, ' Sensitivity: ',report_ffc['recall'], ' AUPRC: ',  auprc_ffc)
    print('AFO - AUC: ',auc_afo, ' Sensitivity: ',report_afo['recall'], ' AUPRC: ',  auprc_afo)
    print('Suresh - AUC: ',auc_suresh, ' Sensitivity: ',report_suresh['recall'], ' AUPRC: ',  auprc_suresh)
    print('Sens - AUC: ',auc_sens, ' Sensitivity: ',report_sens['recall'], ' AUPRC: ',  auprc_sens)
    

    print('---------------------------------------------------thr:', th)
    print('FFC & ', round(auc_ffc,4),  ' & ',  round(auprc_ffc,4) ,'&' , report_ffc['precision'],'&',report_ffc['recall'], '&' ,report_ffc['f1-score'],'\\\\')
    print('AFO &',round(auc_afo,4),  ' & ',  round(auprc_afo,4), '&' , report_afo['precision'],'&', report_afo['recall'],'&',report_afo['f1-score'],'\\\\')
    print('Suresh & ',round(auc_suresh,4),  ' & ',  round(auprc_suresh,4), '&', report_suresh['precision'],'&',report_suresh['recall'],'&',report_suresh['f1-score'],'\\\\')
    print('Sens & ',round(auc_sens,4),  ' & ',  round(auprc_sens,4),'&' , report_sens['precision'] ,'&',report_sens['recall'],'&',report_sens['f1-score'],'\\\\')
    print('Lime & ',round(auc_lime,4),  ' & ',  round(auprc_lime,4),'&' , report_lime['precision'] ,'&',report_lime['recall'],'&',report_lime['f1-score'],'\\\\')
    print('True Gen & ',round(auc_true_gen,4),  ' & ',  round(auprc_true_gen,4),'&' , report_true_gen['precision'] ,'&',report_true_gen['recall'],'&',report_true_gen['f1-score'],'\\\\')
    '''

    auc_ffc.append(auc_ffc_cv)
    auprc_ffc.append(auprc_ffc_cv)
    
    auc_cond.append(auc_cond_cv)
    auprc_cond.append(auprc_cond_cv)

    auc_att.append(auc_att_cv)
    auprc_att.append(auprc_att_cv)

    auc_afo.append(auc_afo_cv)
    auprc_afo.append(auprc_afo_cv)

    auc_fo.append(auc_suresh_cv)
    auprc_fo.append(auprc_suresh_cv)

    auc_sens.append(auc_sens_cv)
    auprc_sens.append(auprc_sens_cv)

    auc_lime.append(auc_lime_cv)
    auprc_lime.append(auprc_lime_cv)

    auc_true_gen.append(auc_true_gen_cv)
    auprc_true_gen.append(auprc_true_gen_cv)
    
print('---------------------------------------------------thr:', th)
print('FFC & ', round(np.mean(auc_ffc),4), '+-' ,round(np.std(auc_ffc),4) ,' & ',  round(np.mean(auprc_ffc),4),'+-',round(np.std(auprc_ffc),4) ,'\\\\')
print('Cond & ', round(np.mean(auc_cond),4), '+-' ,round(np.std(auc_cond),4) ,' & ',  round(np.mean(auprc_cond),4),'+-',round(np.std(auprc_cond),4) ,'\\\\')
print('AFO & ', round(np.mean(auc_afo),4), '+-' ,round(np.std(auc_afo),4) ,' & ',  round(np.mean(auprc_afo),4),'+-',round(np.std(auprc_afo),4) ,'\\\\')
print('FO & ', round(np.mean(auc_fo),4), '+-' ,round(np.std(auc_fo),4) ,' & ',  round(np.mean(auprc_fo),4),'+-',round(np.std(auprc_fo),4) ,'\\\\')
print('Sens & ', round(np.mean(auc_sens),4), '+-' ,round(np.std(auc_sens),4) ,' & ',  round(np.mean(auprc_sens),4),'+-',round(np.std(auprc_sens),4) ,'\\\\')
print('Attention & ', round(np.mean(auc_att),4), '+-' ,round(np.std(auc_att),4) ,' & ',  round(np.mean(auprc_att),4),'+-',round(np.std(auprc_att),4) ,'\\\\')
print('LIME & ', round(np.mean(auc_lime),4), '+-' ,round(np.std(auc_lime),4) ,' & ',  round(np.mean(auprc_lime),4),'+-',round(np.std(auprc_lime),4), '\\\\')
print('True Gen & ', round(np.mean(auc_true_gen),4), '+-' ,round(np.std(auc_true_gen),4) ,' & ',  round(np.mean(auprc_true_gen),4),'+-',round(np.std(auprc_true_gen),4) ,'\\\\')

