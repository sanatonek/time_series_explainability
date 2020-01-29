from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.models import DeepKnn
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer, BaselineExplainer
import torch
import os
import json
import argparse
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score as auc_score
import pickle as pkl
import sys
import glob
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp', 'gender', 'age', 'ethnicity', 'first_icu_stay']
USER = 'sana'


def parse_lime_results(arr,Tt,n_features):
    lime_res = np.zeros((n_features,Tt))
    for t in range(Tt):
        parse_str = np.array(arr['lime']['imp'][0][t][0][0].split(' '))
        feature_idx = np.where(np.array(parse_str)=='feature')[0][0]+1
        feature_val = abs(arr['lime']['imp'][0][t][0][1])
        lime_res[int(parse_str[feature_idx])-1,t]=feature_val
    return lime_res


def choose_scaling_param(data):
    filelist = glob.glob(os.path.join('/scratch/gobi1/%s/TSX_results' % USER, data, 'results_*cv_1.pkl'))

    N = len(filelist)
    with open(filelist[0], 'rb') as f:
        arr = pkl.load(f)

    n_features = arr['FFC']['imp'].shape[0]
    t_len = arr['FFC']['imp'].shape[1]
    y_ffc = np.zeros((N, n_features, t_len))
    gt_importance = np.zeros((N, n_features, t_len))

    for n, file in enumerate(filelist):
        with open(file, 'rb') as f:
            arr = pkl.load(f)
        y_ffc[n, :, :] = arr['FFC']['imp']
        gt_importance[n,:,:] = arr['gt'][:, 1:]

    auc = []
    scaling_params = [0.1, 1, 10, 100, 1e3, 1e5]
    for alpha in scaling_params:
        y_ffc_scaled = 1. / (1. + np.exp(alpha * y_ffc))
        auc.append(auc_score(gt_importance.reshape(-1, ), y_ffc_scaled.reshape(-1, )))

    return scaling_params[np.argmax(auc)], auc


def main(data, alpha):
    auc_ffc = []
    auc_afo = []
    auc_suresh = []
    auc_sens = []
    auc_lime = []
    auprc_ffc = []
    auprc_afo = []
    auprc_suresh = []
    auprc_sens = []
    auprc_lime = []

    for cv in [0,1,2,3,4]:
        filelist = glob.glob(os.path.join('/scratch/gobi1/%s/TSX_results'%USER, data, 'results_*cv_%s.pkl'%str(cv)))

        N = len(filelist)
        with open(filelist[0], 'rb') as f:
            arr = pkl.load(f)

        n_features = arr['FFC']['imp'].shape[0]
        t_len = arr['FFC']['imp'].shape[1]

        y_ffc = np.zeros((N, n_features, t_len))
        y_afo = np.zeros((N, n_features, t_len))
        y_suresh = np.zeros((N, n_features, t_len))
        y_sens = np.zeros((N, n_features, t_len))
        y_lime = np.zeros((N, n_features, t_len))
        gt_importance = np.zeros((N, n_features, t_len))

        for n, file in enumerate(filelist):
            with open(file, 'rb') as f:
                arr = pkl.load(f)

            y_ffc[n, :, :] = 1./(1.+np.exp(alpha*arr['FFC']['imp']))
            y_afo[n, :, :] = arr['AFO']['imp']
            y_suresh[n, :, :] = arr['Suresh_et_al']['imp']
            y_sens[n, :, :] = arr['Sens']['imp'][:, 1:]
            y_lime[n, :, :] = parse_lime_results(arr, t_len, n_features)
            gt_importance[n,:,:] = arr['gt'][:, 1:]

        #print metrics
        auc_ffc.append(metrics.roc_auc_score(gt_importance.reshape(-1,), y_ffc.reshape(-1,)))
        auc_afo.append(metrics.roc_auc_score(gt_importance.reshape(-1,), y_afo.reshape(-1,)))
        auc_suresh.append(metrics.roc_auc_score(gt_importance.reshape(-1,), y_suresh.reshape(-1,)))
        auc_sens.append(metrics.roc_auc_score(gt_importance.reshape(-1,), y_sens.reshape(-1,)))
        auc_lime.append(metrics.roc_auc_score(gt_importance.reshape(-1,), y_lime.reshape(-1,)))
        # auc_true_gen_cv= metrics.roc_auc_score(gt_importance.reshape(-1,), y_true_gen.reshape(-1,))



        # auprc
        auprc_ffc.append(metrics.average_precision_score(gt_importance.reshape(-1,), y_ffc.reshape(-1,)))
        auprc_afo.append(metrics.average_precision_score(gt_importance.reshape(-1,), y_afo.reshape(-1,)))
        auprc_suresh.append(metrics.average_precision_score(gt_importance.reshape(-1,), y_suresh.reshape(-1,)))
        auprc_sens.append(metrics.average_precision_score(gt_importance.reshape(-1,), y_sens.reshape(-1,)))
        auprc_lime.append(metrics.average_precision_score(gt_importance.reshape(-1,), y_lime.reshape(-1,)))
        # auprc_true_gen_cv= metrics.average_precision_score(gt_importance.reshape(-1,), y_true_gen)

    # print('---------------------------------------------------thr:', th)
    print('FFC & ', round(np.mean(auc_ffc), 4), '+-', round(np.std(auc_ffc), 4), ' & ', round(np.mean(auprc_ffc), 4),
          '+-', round(np.std(auprc_ffc), 4), '\\\\')
    print('AFO & ', round(np.mean(auc_afo), 4), '+-', round(np.std(auc_afo), 4), ' & ', round(np.mean(auprc_afo), 4),
          '+-', round(np.std(auprc_afo), 4), '\\\\')
    print('FO & ', round(np.mean(auc_suresh), 4), '+-', round(np.std(auc_suresh), 4), ' & ', round(np.mean(auprc_suresh), 4), '+-',
          round(np.std(auprc_suresh), 4), '\\\\')
    print('Sens & ', round(np.mean(auc_sens), 4), '+-', round(np.std(auc_sens), 4), ' & ',
          round(np.mean(auprc_sens), 4), '+-', round(np.std(auprc_sens), 4), '\\\\')
    print('LIME & ', round(np.mean(auc_lime), 4), '+-', round(np.std(auc_lime), 4), ' & ',
          round(np.mean(auprc_lime), 4), '+-', round(np.std(auprc_lime), 4), '\\\\')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--model', type=str, default='feature_generator_explainer', help='Prediction model')
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--n_features', type=int, default=5)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()
    alpha, auc = choose_scaling_param(args.data)
    print(alpha, auc)
    main(data=args.data, alpha=alpha)
