#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import pickle as pkl
import glob

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
#filelist = glob.glob('/scratch/gobi1/shalmali/'+data+'/results_*.pkl')
filelist = glob.glob('/scratch/gobi1/sana/TSX_results/simulation_non_stationary/results_*.pkl')

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

for cv in range(1):
    #for th in thrs:
    #for n,file in enumerate(filelist):
    y_true=np.zeros(n_features*N*Tt)
    y_ffc=np.zeros(n_features*N*Tt)
    y_afo=np.zeros(n_features*N*Tt)
    y_suresh=np.zeros(n_features*N*Tt)
    y_sens=np.zeros(n_features*N*Tt)
    y_lime=np.zeros(n_features*N*Tt)
    y_true_gen=np.zeros(n_features*N*Tt)

    y_binary_ffc=np.zeros(n_features*N*Tt)
    y_binary_afo=np.zeros(n_features*N*Tt)
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

    for nn, n in enumerate(list(range(0,N))):
    #for nn,file in enumerate(filelist):
        #if cv==0:
        file = glob.glob(os.path.join(fpath,data,'results_'+ str(n) + '.pkl'))[0]
        #print(file)
        #else:
        #print(fpath, data)
        #print(os.path.join(fpath,data,'/results_'+str(n)+ 'cv_'+str(cv)+'.pkl'))
        #file = glob.glob(os.path.join(fpath,data,'results_'+str(n)+ 'cv_'+str(cv)+'.pkl'))[0]
        
        with open(file,'rb') as f:
            arr = pkl.load(f)
        #print(file)
        n_obs=Tt*n_features

        if data!='simulation_spike':
            #print(arr.keys())
            y_true[nn*n_obs:(nn+1)*n_obs] = arr['gt'][:,1:].flatten()
        else:
            gt_array = np.zeros((n_features,Tt))
            gt_array[0,:] = arr['gt'][1:]
            y_true[nn*n_obs:(nn+1)*n_obs] = gt_array.flatten()

        y_ffc[nn*n_obs:(nn+1)*n_obs] = arr['FFC']['imp'].flatten()
        y_afo[nn*n_obs:(nn+1)*n_obs] = arr['AFO']['imp'].flatten()
        y_suresh[nn*n_obs:(nn+1)*n_obs] = arr['Suresh_et_al']['imp'].flatten()
        y_sens[nn*n_obs:(nn+1)*n_obs] = arr['Sens']['imp'][:,1:].flatten()
        y_lime[nn*n_obs:(nn+1)*n_obs] = parse_lime_results(arr,Tt,n_features).flatten()

        #file = glob.glob('/scratch/gobi1/shalmali/'+data+'/results_true_'+str(n)+'.pkl')[0]
        #with open(file,'rb') as f:
        #    arr_true = pkl.load(f)
        #y_true_gen[nn*n_obs:(nn+1)*n_obs] = arr_true.flatten()

        y_binary_ffc[nn*n_obs:(nn+1)*n_obs] = arr['FFC']['imp'].flatten()
        y_binary_ffc[y_binary_ffc>=th] = 1
        y_binary_ffc[y_binary_ffc<th] = 1

        #y_binary_true_gen[nn*n_obs:(nn+1)*n_obs] = arr_true.flatten()
        #y_binary_true_gen[y_binary_true_gen>=th] = 1
        #y_binary_true_gen[y_binary_true_gen<th] = 1

        y_binary_afo[nn*n_obs:(nn+1)*n_obs] = arr['AFO']['imp'].flatten()
        y_binary_afo[y_binary_afo>=th] = 1
        y_binary_afo[y_binary_afo<th] = 1

        y_binary_suresh[nn*n_obs:(nn+1)*n_obs] = arr['Suresh_et_al']['imp'].flatten()
        y_binary_suresh[y_binary_suresh>=th] = 1
        y_binary_suresh[y_binary_suresh<th] = 1

        y_binary_sens[nn*n_obs:(nn+1)*n_obs] = arr['Sens']['imp'][:,1:].flatten()
        y_binary_sens[y_binary_sens>=th] = 1
        y_binary_sens[y_binary_sens<th] = 1

        y_binary_lime[nn*n_obs:(nn+1)*n_obs] = y_lime[nn*n_obs:(nn+1)*n_obs]
        y_binary_lime[y_binary_lime>=th] = 1
        y_binary_lime[y_binary_lime<th] = 1


    #print metrics
    auc_ffc_cv= metrics.roc_auc_score(y_true, y_ffc)
    auc_afo_cv= metrics.roc_auc_score(y_true, y_afo)
    auc_suresh_cv= metrics.roc_auc_score(y_true, y_suresh)
    auc_sens_cv= metrics.roc_auc_score(y_true, y_sens)
    auc_lime_cv= metrics.roc_auc_score(y_true, y_lime)
    auc_true_gen_cv= metrics.roc_auc_score(y_true, y_true_gen)


    # recall/sensitivity
    report_ffc = metrics.classification_report(y_true, y_binary_ffc,output_dict=True)['1.0']
    report_afo = metrics.classification_report(y_true, y_binary_afo,output_dict=True)['1.0']
    report_suresh = metrics.classification_report(y_true, y_binary_suresh,output_dict=True)['1.0']
    report_sens = metrics.classification_report(y_true, y_binary_sens,output_dict=True)['1.0']
    report_lime = metrics.classification_report(y_true, y_binary_lime,output_dict=True)['1.0']
    report_true_gen = metrics.classification_report(y_true, y_binary_true_gen,output_dict=True)['1.0']


    # auprc
    auprc_ffc_cv= metrics.average_precision_score(y_true, y_ffc)
    auprc_afo_cv= metrics.average_precision_score(y_true, y_afo)
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

    precision_ffc.append(report_ffc['precision'])
    recall_ffc.append(report_ffc['recall'])
    auc_ffc.append(auc_ffc_cv)
    auprc_ffc.append(auprc_ffc_cv)

    precision_afo.append(report_afo['precision'])
    recall_afo.append(report_afo['recall'])
    auc_afo.append(auc_afo_cv)
    auprc_afo.append(auprc_afo_cv)

    precision_suresh.append(report_suresh['precision'])
    recall_suresh.append(report_suresh['recall'])
    auc_fo.append(auc_suresh_cv)
    auprc_fo.append(auprc_suresh_cv)

    precision_sens.append(report_sens['precision'])
    recall_sens.append(report_sens['recall'])
    auc_sens.append(auc_sens_cv)
    auprc_sens.append(auprc_sens_cv)

    precision_lime.append(report_lime['precision'])
    recall_lime.append(report_lime['recall'])
    auc_lime.append(auc_lime_cv)
    auprc_lime.append(auprc_lime_cv)

    precision_true_gen.append(report_true_gen['precision'])
    recall_true_gen.append(report_true_gen['recall'])
    auc_true_gen.append(auc_true_gen_cv)
    auprc_true_gen.append(auprc_true_gen_cv)
    
print('---------------------------------------------------thr:', th)
print('FFC & ', round(np.mean(auc_ffc),4), '+-' ,round(np.std(auc_ffc),4) ,' & ',  round(np.mean(auprc_ffc),4),'+-',round(np.std(auprc_ffc),4) ,'\\\\')
print('AFO & ', round(np.mean(auc_afo),4), '+-' ,round(np.std(auc_afo),4) ,' & ',  round(np.mean(auprc_afo),4),'+-',round(np.std(auprc_afo),4) ,'\\\\')
print('FO & ', round(np.mean(auc_fo),4), '+-' ,round(np.std(auc_fo),4) ,' & ',  round(np.mean(auprc_fo),4),'+-',round(np.std(auprc_fo),4) ,'\\\\')
print('Sens & ', round(np.mean(auc_sens),4), '+-' ,round(np.std(auc_sens),4) ,' & ',  round(np.mean(auprc_sens),4),'+-',round(np.std(auprc_sens),4) ,'\\\\')
print('LIME & ', round(np.mean(auc_lime),4), '+-' ,round(np.std(auc_lime),4) ,' & ',  round(np.mean(auprc_lime),4),'+-',round(np.std(auprc_lime),4), '\\\\')
print('True Gen & ', round(np.mean(auc_true_gen),4), '+-' ,round(np.std(auc_true_gen),4) ,' & ',  round(np.mean(auprc_true_gen),4),'+-',round(np.std(auprc_true_gen),4) ,'\\\\')


