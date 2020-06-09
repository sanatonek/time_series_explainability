import os
import numpy as np
import pickle as pkl
from sklearn import metrics
import argparse


def performance_metric(score, g_truth):
    n = len(score)
    Temp_TPR = np.zeros([n, ])
    Temp_FDR = np.zeros([n, ])
    score = np.clip(score, 0, 1)

    for i in range(n):
        # TPR
        # print(score[i, :10], g_truth[i, :10])
        TPR_Nom = np.dot(score[i, :], (g_truth[i, :]))#np.sum(score[i, :] * g_truth[i, :])
        TPR_Den = np.sum(g_truth[i, :])
        # print(float(TPR_Nom), float(TPR_Den + 1e-18))
        Temp_TPR[i] = float(TPR_Nom) / float(TPR_Den + 1e-18)

        # FDR
        FDR_Nom = np.dot(score[i, :], (1 - g_truth[i, :]))#np.sum(score[i, :] * (1 - g_truth[i, :]))
        FDR_Den = np.sum(score[i, :])
        Temp_FDR[i] = float(FDR_Nom) / float(FDR_Den + 1e-18)

    return np.mean(Temp_TPR), np.mean(Temp_FDR), np.std(Temp_TPR), np.std(Temp_FDR)

def main(args):
    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
        data_type = 'state'
    elif args.data == 'simulation_l2x':
        feature_size = 3
        data_path = './data/simulated_data_l2x'
        data_type = 'state'
    elif args.data == 'simulation_spike':
        feature_size = 3
        data_path = './data/simulated_spike_data'
        data_type = 'spike'

    score_path = '/scratch/gobi1/shalmali/TSX_results/new_results/%s' %(args.data)
    if data_type == 'state':
        with open(os.path.join(data_path, 'state_dataset_importance_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)
    elif data_type == 'spike':
        with open(os.path.join(data_path, 'gt_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)

    auc, aupr, fdr, tpr = [], [], [], []
    for cv in [0, 1, 2, 3, 4]:
        with open(os.path.join(score_path, '%s_test_importance_scores_%s.pkl' %(args.explainer, str(cv))), 'rb') as f:
            importance_scores = pkl.load(f)

        gt_importance_test.astype(int)
        gt_score = gt_importance_test.flatten()
        explainer_score = importance_scores.flatten()
        n = len(gt_importance_test)
        if (args.explainer == 'deep_lift' or args.explainer == 'integrated_gradient' or args.explainer == 'gradient_shap'):
            explainer_score = np.abs(explainer_score)
        auc_score = metrics.roc_auc_score(gt_score, explainer_score)
        aupr_score = metrics.average_precision_score(gt_score, explainer_score)
        p_metric = performance_metric(importance_scores.reshape(n,-1), gt_importance_test.reshape(n,-1))
        tpr.append(p_metric[0])
        fdr.append(p_metric[1])
        auc.append(auc_score)
        aupr.append(aupr_score)
    print(args.explainer, ' auc: %.3f +- %.3f'%(np.mean(auc), np.std(auc)), ' aupr: %.3f +- %.3f'%(np.mean(aupr), np.std(aupr)))
    print(args.explainer, ' tpr: %.3f +- %.3f' % (np.mean(tpr), np.std(tpr)), ' fdr: %.3f +- %.3f' % (np.mean(fdr), np.std(fdr)))


if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--generator_type', type=str, default='history')
    args = parser.parse_args()
    main(args)