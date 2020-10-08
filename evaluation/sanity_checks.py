from TSX.utils import load_data, load_simulated_data, test_model_rt, train_model
from TSX.experiments import FeatureGeneratorExplainer, BaselineExplainer

import torch
import os
import sys
import argparse
import pickle as pkl
import numpy as np
from scipy.signal import correlate as correlate
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# from baseline_results import parse_lime_results
from TSX.models import StateClassifier, RETAIN
from TSX.generator import JointFeatureGenerator
from TSX.explainers import FITExplainer, GradientShapExplainer, FOExplainer, AFOExplainer, DeepLiftExplainer, \
    IGExplainer, RETAINexplainer
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp']

MIMIC_TEST_SAMPLES = [1534, 3734, 82, 3663, 3509, 870, 3305, 1484, 2604, 1672, 2733, 1057, 2599, 3319, 1239, 1671, 3095, 3783, 1935, 720, 1961, 3476, 262, 816, 2268, 723, 4469, 3818, 4126, 1575, 1526, 1457, 4542, 2015, 2512, 1419, 1749, 3822, 466, 165, 306, 1922, 1973, 1218, 1987, 701, 3344, 2285, 2363, 1429, 808, 3266, 3643, 8, 4528, 156, 229, 2684, 3588, 532, 436, 2934, 503, 2635, 4077, 2112, 2776, 2012, 2724, 420, 2978, 4265, 832, 309, 3748, 1260, 1294, 1423, 2787, 1012, 2177, 1335, 53, 2054, 2135, 4266, 3379, 379, 1580, 1720, 4409, 415, 4273, 3927, 3226, 2316, 1933, 3442, 3047, 1219, 1308, 614, 3115, 1237, 2191, 838, 3367, 1751, 2362, 3180, 2800, 2871, 3168, 3839, 4153, 7, 1014, 4428, 1803, 766, 494, 3184, 3179, 2004, 3450, 3586, 2460, 429, 1547, 1630, 1586, 4090, 2781, 2108, 1849, 4278, 2820, 1799, 1936, 1895, 1741, 4015, 3373, 973, 2291, 3122, 2979, 3377, 3892, 3742, 508, 155, 1122, 1919, 708, 1909, 3950, 4236, 3797, 4403, 4220, 3779, 1954, 3754, 3174, 2850, 2303, 1375, 1431, 67, 1278, 1757, 789, 3716, 2666, 2145, 658, 2270, 3829, 3600, 811, 3334, 79, 3803, 4131, 300, 3026, 2013, 3064, 4369, 1174, 1857, 55, 3156, 2732, 1573, 4423, 3856, 2882, 831, 2933, 3325, 1994, 440, 3788, 3126, 434, 1777, 3717, 3067, 4253, 4301, 3380, 4584, 1307, 1043, 1786, 670, 3644, 1524, 4489, 1886, 3258, 1115, 2394, 3008, 364, 4065, 3515, 2348, 2141, 3060, 1642, 1868, 4575, 4271, 967, 834, 1906, 2836, 2138, 3641, 30, 1611, 4360, 3472, 3117, 3732, 1728, 4537, 3154, 4513, 4474, 453, 3002, 4103, 4348, 2745, 3510, 299, 2157, 2718, 4127, 1811, 2523, 261, 4337, 2541, 2244, 3158, 1236, 589, 285, 2445, 4413, 893, 2272, 2422, 3639, 4556, 1067, 907, 350, 2491, 3841, 3876, 3921, 2117, 39, 2788, 179, 3314, 1083, 2038, 1776, 4356, 2926, 3786, 3323, 147]

SIMULATION_SAMPLES = np.random.choice(100, 10)
samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}


def main(args):
    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
        data_type='state'
    elif args.data == 'simulation_l2x':
        feature_size = 3
        data_path = './data/simulated_data_l2x'
        data_type='state'
    elif args.data == 'simulation_spike':
        feature_size = 3
        data_path = './data/simulated_spike_data'
        data_type='spike'
    elif args.data == 'mimic':
        data_type = 'mimic'
        timeseries_feature_size = len(feature_map_mimic)

    # Load data
    if args.data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path='./data', cv=args.cv)
        feature_size = p_data.feature_size
    else:
        _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, datapath=data_path,
                                                                         percentage=0.8, data_type=data_type)


    model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=200)

    if args.explainer == 'fit':
        generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)
        generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'joint_generator'))))


    testset = [smpl[0] for smpl in test_loader.dataset]
    samples = torch.stack([testset[sample] for sample in samples_to_analyze[args.data]])

    model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))
    if args.explainer == 'fit':
        explainer = FITExplainer(model, generator)
    elif args.explainer == 'integrated_gradient':
        explainer = IGExplainer(model)
    elif args.explainer == 'deep_lift':
        explainer = DeepLiftExplainer(model)
    elif args.explainer == 'fo':
        explainer = FOExplainer(model)
    elif args.explainer == 'afo':
        explainer = AFOExplainer(model, train_loader)
    elif args.explainer == 'gradient_shap':
        explainer = GradientShapExplainer(model)
    elif args.explainer == 'retain':
        model = RETAIN(dim_input=feature_size, dim_emb=128, dropout_emb=0.4, dim_alpha=8, dim_beta=8,
                       dropout_context=0.4, dim_output=2)
        explainer = RETAINexplainer(model, args.data)
        model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'retain'))))
    gt_importance = explainer.attribute(samples, torch.zeros(samples.shape))

    for r_ind, ratio in enumerate([.2, .4, .6, .8, 1.]):
        for param in model.parameters():
            params = param.data.cpu().numpy().reshape(-1)
            params[int(r_ind * 0.2):int(ratio * len(params))] = torch.randn(int(ratio * len(params)))
            param.data = torch.Tensor(params.reshape(param.data.shape))
        if args.explainer == 'fit':
            explainer = FITExplainer(model, generator)
        elif args.explainer == 'integrated_gradient':
            explainer = IGExplainer(model)
        elif args.explainer == 'deep_lift':
            explainer = DeepLiftExplainer(model)
        elif args.explainer == 'fo':
            explainer = FOExplainer(model)
        elif args.explainer == 'afo':
            explainer = AFOExplainer(model, train_loader)
        elif args.explainer == 'gradient_shap':
            explainer = GradientShapExplainer(model)
        elif args.explainer == 'retain':
            model = RETAIN(dim_input=feature_size, dim_emb=128, dropout_emb=0.4, dim_alpha=8, dim_beta=8,
                           dropout_context=0.4, dim_output=2)
            explainer = RETAINexplainer(model, args.data)
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'retain'))))

        score = explainer.attribute(samples, torch.zeros(samples.shape))
        corr = []
        for sig in range(len(score)):
            corr.append(abs(spearmanr(score[sig].reshape(-1,), gt_importance[sig].reshape(-1,), nan_policy='omit')[0]))
        print("correlation for %d percent randomization: %.3f +- %.3f" % (100*ratio, np.mean(corr), np.std(corr)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple sanity checks on the data')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--check', type=str, default='randomized_param')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--all_samples', action='store_true')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--cv', type=int, default=0)
    args = parser.parse_args()
    main(args)
