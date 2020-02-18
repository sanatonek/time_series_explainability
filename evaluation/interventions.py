import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import torch

from TSX.utils import load_data

import pickle as pkl
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

USER = "sana"

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp']
color_map = ['#00998F', '#C20088', '#0075DC', '#E0FF66', '#4C005C', '#191919', '#FF0010', '#2BCE48', '#FFCC99', '#808080',
             '#740AFF', '#8F7C00', '#9DCC00', '#F0A3FF', '#94FFB5', '#FFA405', '#FFA8BB', '#426600', '#005C31', '#5EF1F2',
             '#993F00', '#990000', '#003380', '#990000', '#FFFF80', '#FF5005', '#FFFF00','#FF0010', '#FFCC99']


def summary_stat(intervention_ID, patient_data, cv=2):
    interventions = patient_data.test_intervention[:, intervention_ID, :]
    df = pd.DataFrame(columns=['pid', 'intervention_id', 'method', 'top1', 'top2', 'top3'])
    if hasattr(patient_data, 'test_intervention'):
        ind_list = np.where(np.sum(interventions[:, 1:], axis=1) != 0)[0]  ## Index of subject that have intervention=intervention_ID data recorded

        for ind, subject in enumerate(ind_list):
            if subject > 70:
                continue
            intervention = interventions[subject, 1:]
            start_point = np.argwhere(intervention == 1)[0][0]

            # print(start_point)
            if start_point < 10:
                continue
            max_imp_FCC = []
            max_imp_occ = []
            max_imp_sen = []
            max_imp_afo = []

            with open(os.path.join('/scratch/gobi1/%s/TSX_results/mimic'%USER, 'results_%scv_%d.pkl'%(subject, cv)),'rb') as f:
                arr = pkl.load(f)
            importance = arr['FFC']['imp'][:, :start_point + 1]
            importance_afo = arr['AFO']['imp'][:, :start_point + 1]
            importance_occ = arr['Suresh_et_al']['imp'][:, :start_point + 1]
            sensitivity_analysis = arr['Sens']['imp'][:, :start_point + 1]

            for i in range(27):
                max_imp_FCC.append((i, max(importance[i, :])))
                max_imp_occ.append((i, max(importance_occ[i, :])))
                max_imp_afo.append((i, max(importance_afo[i, :])))
                max_imp_sen.append((i, sensitivity_analysis[i, start_point]))

            max_imp_FCC.sort(key=lambda pair: pair[1], reverse=True)
            max_imp_occ.sort(key=lambda pair: pair[1], reverse=True)
            max_imp_sen.sort(key=lambda pair: pair[1], reverse=True)
            max_imp_afo.sort(key=lambda pair: pair[1], reverse=True)

            df.loc[-1] = [subject, intervention_ID, 'FCC', max_imp_FCC[0][0], max_imp_FCC[1][0],
                          max_imp_FCC[2][0]]  # adding a row
            df.index = df.index + 1
            df.loc[-1] = [subject, intervention_ID, 'f_occ', max_imp_occ[0][0], max_imp_occ[1][0],
                          max_imp_occ[2][0]]  # adding a row
            df.index = df.index + 1
            df.loc[-1] = [subject, intervention_ID, 'afo', max_imp_afo[0][0], max_imp_afo[1][0],
                          max_imp_afo[2][0]]  # adding a row
            df.index = df.index + 1
            df.loc[-1] = [subject, intervention_ID, 'sensitivity', max_imp_sen[0][0], max_imp_sen[1][0],
                          max_imp_sen[2][0]]  # adding a row
            df.index = df.index + 1
        if not os.path.exists(os.path.join('/scratch/gobi1/%s/TSX_results/mimic'%USER,'interventions')):
            os.mkdir(os.path.join('/scratch/gobi1/%s/TSX_results/mimic'%USER,'interventions'))
        df.to_pickle(os.path.join('/scratch/gobi1/%s/TSX_results/mimic'%USER,'interventions/int_%d.pkl' % (intervention_ID)))


def plot_summary_stat(intervention_ID):
    df = pd.read_pickle(os.path.join('/scratch/gobi1/%s/TSX_results/mimic'%USER,'interventions/int_%d.pkl' % (intervention_ID)))
    fcc_df = df.loc[df['method'] == 'FCC']
    afo_df = df.loc[df['method'] == 'afo']
    occ_df = df.loc[df['method'] == 'f_occ']
    sen_df = df.loc[df['method'] == 'sensitivity']
    fcc_dist = np.sort(np.array(fcc_df[['top1', 'top2', 'top3']]).reshape(-1, ))
    afo_dist = np.sort(np.array(afo_df[['top1', 'top2', 'top3']]).reshape(-1, ))
    occ_dist = np.sort(np.array(occ_df[['top1', 'top2', 'top3']]).reshape(-1, ))
    sen_dist = np.sort(np.array(sen_df[['top1', 'top2', 'top3']]).reshape(-1, ))


    fcc_top = create_pairs(find_count(fcc_dist))[0:4]
    afo_top = create_pairs(find_count(afo_dist))[0:4]
    occ_top = create_pairs(find_count(occ_dist))[0:4]
    sen_top = create_pairs(find_count(sen_dist))[0:4]
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey=True)
    ax1.bar([feature_map_mimic[x[0]] for x in fcc_top], [x[1] for x in fcc_top],
            color=[color_map[x[0]] for x in fcc_top])
    ax2.bar([feature_map_mimic[x[0]] for x in afo_top], [x[1] for x in afo_top],
            color=[color_map[x[0]] for x in afo_top])
    ax3.bar([feature_map_mimic[x[0]] for x in occ_top], [x[1] for x in occ_top],
            color=[color_map[x[0]] for x in occ_top])
    ax4.bar([feature_map_mimic[x[0]] for x in sen_top], [x[1] for x in sen_top],
            color=[color_map[x[0]] for x in sen_top])
    f.suptitle('%s' % (intervention_list[intervention_ID]), fontweight='bold', fontsize=36)
    ax1.set_title('FFC', fontsize=32, fontweight='bold')
    ax2.set_title('AFO', fontsize=32, fontweight='bold')
    ax3.set_title('FO', fontsize=32, fontweight='bold')
    ax4.set_title('Sensitivity analysis', fontsize=32, fontweight='bold')
    ax1.tick_params(labelsize=26)
    ax2.tick_params(labelsize=26)
    ax3.tick_params(labelsize=26)
    ax4.tick_params(labelsize=26)
    plt.subplots_adjust(hspace=0.6)
    f.set_figheight(12)
    f.set_figwidth(15)
    if not os.path.exists('./plots/distributions'):
        os.mkdir('./plots/distributions')
    plt.savefig('./plots/distributions/top_%s.pdf' % (intervention_list[intervention_ID]), dpi=300, bbox_inches='tight')

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.bar(feature_map_mimic, find_count(fcc_dist))
    ax2.bar(feature_map_mimic, find_count(occ_dist))
    ax3.bar(feature_map_mimic, find_count(sen_dist))
    ax1.set_title('FFC importance distribution for %s' % (intervention_list[intervention_ID]), fontsize=20)
    ax2.set_title('feature occlusion importance distribution for %s' % (intervention_list[intervention_ID]),
                  fontsize=20)
    ax3.set_title('sensitivity analysis importance distribution for %s' % (intervention_list[intervention_ID]),
                  fontsize=20)
    f.set_figheight(20)
    f.set_figwidth(15)
    plt.savefig('./plots/distributions/%s.pdf' % (intervention_list[intervention_ID]))


def create_pairs(a):
    l = []
    for i, element in enumerate(a):
        l.append((i, element))
    l.sort(key=lambda x: x[1], reverse=True)
    return l


def find_count(a):
    count_arr = np.zeros(len(feature_map_mimic), )
    for elem in a:
        count_arr[elem] += 1
    return count_arr


def find_closest(arr, target):
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
                return get_closest(arr[mid - 1], arr[mid], target)
            j = mid
        else:
            if mid < n - 1 and target < arr[mid + 1]:
                return get_closest(arr[mid], arr[mid + 1], target)
            i = mid + 1
    return arr[mid]


def get_closest(val1, val2, target):
    if target - val1 >= val2 - target:
        return val2
    else:
        return val1


def main(args):
    print('********** Intervention Experiment **********' )
    p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path='./data', cv=args.cv)
    for id in range(len(intervention_list)):
        summary_stat(intervention_ID=id, patient_data=p_data, cv=args.cv)
        plot_summary_stat(id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract population level information about the interventions')
    parser.add_argument('--cv', type=int, default=0)
    args = parser.parse_args()
    main(args)
