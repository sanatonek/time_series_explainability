#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import pickle as pkl


# In[101]:


#preprocess before metric collection
Tt=79
N=199
y_true=np.zeros(N*Tt)
y_ffc=np.zeros(N*Tt)
y_afo=np.zeros(N*Tt)
y_suresh=np.zeros(N*Tt)
y_sens=np.zeros(N*Tt)

y_binary_ffc=np.zeros(N*Tt)
y_binary_afo=np.zeros(N*Tt)
y_binary_suresh=np.zeros(N*Tt)
y_binary_sens=np.zeros(N*Tt)

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


for th in thrs:
    for n in range(nn):
        with open('../examples/simulation/results_'+str(n)+'.pkl','rb') as f:
            arr = pkl.load(f)
        y_true[n*Tt:(n+1)*Tt] = arr['gt'][1:]
        y_ffc[n*Tt:(n+1)*Tt] = arr['FFC'][0,:]
        y_afo[n*Tt:(n+1)*Tt] = arr['AFO'][0,:]
        y_suresh[n*Tt:(n+1)*Tt] = arr['Suresh_et_al'][0,:]
        y_sens[n*Tt:(n+1)*Tt] = arr['Sens'][0,1:]

        y_binary_ffc[n*Tt:(n+1)*Tt] = np.argmax(arr['FFC'],axis=0)
        y_binary_ffc[n*Tt:(n+1)*Tt] = (np.logical_and(y_binary_ffc[n*Tt:(n+1)*Tt]==0,np.max(arr['FFC'],axis=0)>th)).astype(int)
        
        y_binary_afo[n*Tt:(n+1)*Tt] = np.argmax(arr['AFO'],axis=0)
        y_binary_afo[n*Tt:(n+1)*Tt] = (np.logical_and(y_binary_afo[n*Tt:(n+1)*Tt]==0 , np.max(arr['AFO'],axis=0)>th)).astype(int)


        y_binary_suresh[n*Tt:(n+1)*Tt] = np.argmax(arr['Suresh_et_al'],axis=0)
        y_binary_suresh[n*Tt:(n+1)*Tt] = (np.logical_and(y_binary_suresh[n*Tt:(n+1)*Tt]==0 ,np.max(arr['Suresh_et_al'],axis=0)>th)).astype(int)


        y_binary_sens[n*Tt:(n+1)*Tt] = np.argmax(arr['Sens'][:,1:],axis=0)
        y_binary_sens[n*Tt:(n+1)*Tt] = (np.logical_and(y_binary_sens[n*Tt:(n+1)*Tt]==0 ,np.max(arr['Sens'][:,1:],axis=0)>th)).astype(int)

     
    #print metrics
    auc_ffc= metrics.roc_auc_score(y_true, y_ffc)
    auc_afo= metrics.roc_auc_score(y_true, y_afo)
    auc_suresh= metrics.roc_auc_score(y_true, y_suresh)
    auc_sens= metrics.roc_auc_score(y_true, y_sens)
        

    # recall/sensitivity
    report_ffc = metrics.classification_report(y_true, y_binary_ffc,output_dict=True)['1.0']
    report_afo = metrics.classification_report(y_true, y_binary_afo,output_dict=True)['1.0']
    report_suresh = metrics.classification_report(y_true, y_binary_suresh,output_dict=True)['1.0']
    report_sens = metrics.classification_report(y_true, y_binary_sens,output_dict=True)['1.0']
        
    # auprc
    auprc_ffc= metrics.average_precision_score(y_true, y_ffc)
    auprc_afo= metrics.average_precision_score(y_true, y_afo)
    auprc_suresh= metrics.average_precision_score(y_true, y_suresh)
    auprc_sens= metrics.average_precision_score(y_true, y_sens)

    '''
    print('FFC - AUC: ',auc_ffc, ' Sensitivity: ',report_ffc['recall'], ' AUPRC: ',  auprc_ffc)
    print('AFO - AUC: ',auc_afo, ' Sensitivity: ',report_afo['recall'], ' AUPRC: ',  auprc_afo)
    print('Suresh - AUC: ',auc_suresh, ' Sensitivity: ',report_suresh['recall'], ' AUPRC: ',  auprc_suresh)
    print('Sens - AUC: ',auc_sens, ' Sensitivity: ',report_sens['recall'], ' AUPRC: ',  auprc_sens)
    '''

    print('---------------------------------------------------thr:', th)
    print('FFC & ', round(auc_ffc,4),  ' & ',  round(auprc_ffc,4) ,'&' , report_ffc['precision'],'&',report_ffc['recall'], '&' ,report_ffc['f1-score'],'\\\\')
    print('AFO &',round(auc_afo,4),  ' & ',  round(auprc_afo,4), '&' , report_afo['precision'],'&', report_afo['recall'],'&',report_afo['f1-score'],'\\\\')
    print('Suresh & ',round(auc_suresh,4),  ' & ',  round(auprc_suresh,4), '&', report_suresh['precision'],'&',report_suresh['recall'],'&',report_suresh['f1-score'],'\\\\')
    print('Sens & ',round(auc_sens,4),  ' & ',  round(auprc_sens,4),'&' , report_sens['precision'] ,'&',report_sens['recall'],'&',report_sens['f1-score'],'\\\\')

    precision_ffc.append(report_ffc['precision'])
    recall_ffc.append(report_ffc['recall'])
    
    precision_afo.append(report_afo['precision'])
    recall_afo.append(report_afo['recall'])
    
    precision_suresh.append(report_suresh['precision'])
    recall_suresh.append(report_suresh['recall'])
    
    precision_sens.append(report_sens['precision'])
    recall_sens.append(report_sens['recall'])

print('aupr ffc - final :', metrics.auc(precision_ffc,recall_ffc,thrs))
print('aupr afo - final :', metrics.auc(precision_afo,recall_afo,thrs))
#print('aupr suresh - final :', metrics.auc(precision_suresh,recall_suresh,thrs))
print('aupr sens - final :', metrics.auc(precision_sens,recall_sens,thrs))

