from TSX.utils import load_data, load_simulated_data, load_ghg_data, test_model_rt, train_model
from TSX.experiments import FeatureGeneratorExplainer, BaselineExplainer

import torch
import os
import sys
import json
import argparse
import pickle as pkl
import numpy as np
from scipy.signal import correlate as correlate
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from baseline_results import parse_lime_results
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp']

MIMIC_TEST_SAMPLES = [3095, 1971, 1778, 1477, 3022, 8, 262, 3437, 1534, 619, 2076, 1801, 4006, 6, 1952, 2582, 4552,
                       1645, 3324, 1821, 323, 954, 2589, 4449, 1057, 2531, 2733, 2871, 316, 3416, 2110, 3663, 3305,
                       1422, 4126, 3623, 2563, 870, 2717, 1620, 3586, 1683, 1994, 3734, 1671, 4010, 1575, 2254, 588,
                       2838, 1171, 2604, 2702, 3500, 80, 3824, 1672, 1526, 3713, 1527, 1411, 343, 82, 2644, 4199, 3055,
                       4007, 1138, 3087, 606, 1624, 1103, 816, 946, 3435, 1959, 581, 1235, 2238, 2964, 924, 3509, 4385,
                       345, 4061, 1310, 2553, 4483, 3589, 4517, 2799, 1991, 2599, 2745, 4564, 3569, 3484, 897, 304]
SIMULATION_SAMPLES = np.random.choice(100, 10)
samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}


def main(train, data, all_samples, check):
    experiment = "saliency test"
    batch_size = 100
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    generator_type = 'joint_RNN_generator'
    print('********** %s Experiment **********' % (experiment))
    with open('config.json') as config_file:
        configs = json.load(config_file)[data]["feature_generator_explainer"]

    # Load the data
    if data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=batch_size,
                                                                    path='./data')
        feature_size = p_data.feature_size
        # samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}
    elif data == 'ghg':
        p_data, train_loader, valid_loader, test_loader = load_ghg_data(batch_size)
        feature_size = p_data.feature_size
    elif data == 'simulation_spike':
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=batch_size,
                                                                              path='./data_generator/data/simulated_data',
                                                                              data_type='spike')
        feature_size = p_data.shape[1]

    elif data == 'simulation':
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=batch_size,
                                                                              path='./data/simulated_data')
        feature_size = p_data.shape[1]

    if data == 'simulation_spike':
        data = 'simulation'
        spike_data = True
    else:
        spike_data = False


    ## Sanity checks
    if check=='randomized_data':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                        generator_hidden_size=configs['encoding_size'], prediction_size=1,
                                        historical=(configs['historical'] == 1),
                                        generator_type=generator_type, data=data,
                                        experiment=experiment + '_' + generator_type, spike_data=spike_data)
        trainset = list(train_loader.dataset)
        shuffled_labels = torch.stack([sample[1] for sample in trainset])
        r = torch.randperm(len(shuffled_labels))
        shuffled_labels = shuffled_labels[r]
        signals = torch.stack([sample[0] for sample in trainset])
        shuffled_trainset = torch.utils.data.TensorDataset(signals, shuffled_labels)
        shuffled_trainloader = torch.utils.data.DataLoader(shuffled_trainset, batch_size=100)
        exp.risk_predictor.train()
        print("Training model on shuffled data ...")
        opt = torch.optim.Adam(exp.risk_predictor.parameters(), lr=0.0001, weight_decay=1e-3)
        train_model(exp.risk_predictor, shuffled_trainloader, valid_loader, opt, 100, device, data=data)
        exp.risk_predictor.eval()
        _, _, precision, auc, correct_label = test_model_rt(exp.risk_predictor, exp.test_loader)
        with open(os.path.join('./data/simulated_data/state_dataset_importance_test.pkl'), 'rb') as f:
            gt_importance = pkl.load(f)

        ## Sensitivity analysis as a baseline
        testset = [smpl[0] for smpl in exp.test_loader.dataset]
        signal = torch.stack([testset[sample] for sample in samples_to_analyze[exp.data]])
        tvec = list(range(1, signal.shape[2] + 1))
        sensitivity_analysis = np.zeros((signal.shape))
        exp.risk_predictor.train()
        for t_ind, t in enumerate(tvec):
            # print(t)
            signal_t = torch.Tensor(signal[:, :, :t]).to(exp.device).requires_grad_()
            out = exp.risk_predictor(signal_t)
            for s in range(len(samples_to_analyze[exp.data])):
                out[s].backward(retain_graph=True)
                sensitivity_analysis[s, :, t_ind] = signal_t.grad.data[s, :, t_ind].cpu().detach().numpy()  # [:,0]
            signal_t.grad.data.zero_()
        exp.risk_predictor.eval()

        all_FFC_importance = []
        all_FO_importance = []
        all_AFO_importance = []
        FFC_importance = []
        AFO_importance = []
        FO_importance = []
        sensitivity_analysis_importance = []

        signals_to_analyze = range(0, 3)
        for sub_ind, sample_ID in enumerate(samples_to_analyze[exp.data]):
            top_FCC, importance, top_occ, importance_occ, top_occ_aug, importance_occ_aug, top_SA, importance_SA = \
                exp.plot_baseline(sample_ID, signals_to_analyze, sensitivity_analysis[sub_ind, :, :], data=exp.data,
                                  plot=False,
                                  gt_importance_subj=None, tvec=tvec, cv=50)
            all_FFC_importance.append(importance)
            all_AFO_importance.append(importance_occ_aug)
            all_FO_importance.append(importance_occ)

            with open(os.path.join(
                    '/scratch/gobi1/sana/TSX_results/simulation_cv_results/results_%scv_0.pkl' % str(sample_ID)),
                      'rb') as f:
                arr = pkl.load(f)
            FFC_importance.append(arr['FFC']['imp'])
            AFO_importance.append(arr['AFO']['imp'])
            FO_importance.append(arr['Suresh_et_al']['imp'])
            sensitivity_analysis_importance.append(arr['Sens']['imp'])

        FFC_importance_rndm = np.array(all_FFC_importance)
        AFO_importance_rndm = np.array(all_AFO_importance)
        FO_importance_rndm = np.array(all_FO_importance)
        sensitivity_analysis_rndm = np.array(sensitivity_analysis)
        FFC_importance = np.array(FFC_importance)
        AFO_importance = np.array(AFO_importance)
        FO_importance = np.array(FO_importance)
        sensitivity_analysis_importance = np.array(sensitivity_analysis_importance)

        FFC_importance_rndm = np.nan_to_num(FFC_importance_rndm)
        AFO_importance_rndm = np.nan_to_num(AFO_importance_rndm)
        FO_importance_rndm = np.nan_to_num(FO_importance_rndm)
        sensitivity_analysis_rndm = np.nan_to_num(sensitivity_analysis_rndm)

        FFC_corr = []
        AFO_corr = []
        FO_corr = []
        sens_corr = []
        for sig in range(len(FFC_importance)):
            FFC_corr.append(spearmanr(FFC_importance_rndm[sig].reshape(-1, ), FFC_importance[sig].reshape(-1, ))[0])
            AFO_corr.append(spearmanr(AFO_importance_rndm[sig].reshape(-1, ), AFO_importance[sig].reshape(-1, ))[0])
            FO_corr.append(spearmanr(FO_importance_rndm[sig].reshape(-1, ), FO_importance[sig].reshape(-1, ))[0])
            sens_corr.append(spearmanr(sensitivity_analysis_rndm[sig].reshape(-1, ),
                                       sensitivity_analysis_importance[sig].reshape(-1, ))[0])

        FFC_distance = []
        FO_distance = []
        AFO_distance = []
        sensitivity_distance = []
        for s in range(len(FFC_importance)):
            FFC_distance.append(np.linalg.norm((FFC_importance[s] - FFC_importance_rndm[s])))
            FO_distance.append(np.linalg.norm((FO_importance[s] - FO_importance_rndm[s])))
            AFO_distance.append(np.linalg.norm((AFO_importance[s] - AFO_importance_rndm[s])))
            sensitivity_distance.append(
                np.linalg.norm((sensitivity_analysis_importance[s] - sensitivity_analysis_rndm[s])))

        print("FFC correlation: %.3f +- %.3f" % (np.mean(FFC_corr), np.std(FFC_corr)))
        print("AFO correlation: %.3f +- %.3f" % (np.mean(AFO_corr), np.std(AFO_corr)))
        print("FO correlation: %.3f +- %.3f" % (np.mean(FO_corr), np.std(FO_corr)))
        print("Sensitivity correlation: %.3f +- %.3f" % (np.mean(sens_corr), np.std(sens_corr)))

        print("FFC l2 distance: %.3f +- %.3f" % (np.mean(np.array(FFC_distance)), np.std(np.array(FFC_distance))))
        print("AFO l2 distance: %.3f +- %.3f" % (np.mean(np.array(AFO_distance)), np.std(np.array(AFO_distance))))
        print("FO l2 distance: %.3f +- %.3f" % (np.mean(np.array(FO_distance)), np.std(np.array(FO_distance))))
        print("Sensitivity l2 distance: %.3f +- %.3f" % (
            np.mean(np.array(sensitivity_distance)), np.std(np.array(sensitivity_distance))))

    elif check=='randomized_param':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                            generator_hidden_size=configs['encoding_size'], prediction_size=1,
                                            historical=(configs['historical'] == 1),
                                            generator_type=generator_type, data=data,
                                            experiment=experiment + '_' + generator_type, spike_data=spike_data)
        FFC_corr_all = []
        AFO_corr_all = []
        FO_corr_all = []
        Sens_corr_all = []
        for r_ind, ratio in enumerate([.2, .4, .6, .8, 1.]):
            exp.risk_predictor.load_state_dict(torch.load(os.path.join(exp.ckpt_path, 'risk_predictor_RNN.pt')))
            exp.risk_predictor = exp.risk_predictor.to(exp.device)
            exp.risk_predictor.eval()
            for param in exp.risk_predictor.parameters():
                params = param.data.cpu().numpy().reshape(-1)
                params[int(r_ind*0.2):int(ratio*len(params))] = torch.randn(int(ratio*len(params)))
                param.data = torch.Tensor(params.reshape(param.data.shape)).to(exp.device)
            _, _, precision, auc, correct_label = test_model_rt(exp.risk_predictor, exp.test_loader)
            with open(os.path.join('./data/simulated_data/state_dataset_importance_test.pkl'), 'rb') as f:
                gt_importance = pkl.load(f)
            ## Sensitivity analysis as a baseline
            testset = [smpl[0] for smpl in exp.test_loader.dataset]
            signal = torch.stack([testset[sample] for sample in samples_to_analyze[exp.data]])
            tvec = list(range(1, signal.shape[2] + 1))
            sensitivity_analysis = np.zeros((signal.shape))
            exp.risk_predictor.train()
            for t_ind, t in enumerate(tvec):
                    # print(t)
                signal_t = torch.Tensor(signal[:, :, :t]).to(exp.device).requires_grad_()
                out = exp.risk_predictor(signal_t)
                for s in range(len(samples_to_analyze[exp.data])):
                    out[s].backward(retain_graph=True)
                    sensitivity_analysis[s, :, t_ind] = signal_t.grad.data[s, :, t_ind].cpu().detach().numpy()  # [:,0]
                signal_t.grad.data.zero_()
            exp.risk_predictor.eval()

            all_FFC_importance = []
            all_FO_importance = []
            all_AFO_importance = []
            all_lime_importance = []
            FFC_importance = []
            AFO_importance = []
            FO_importance = []
            lime_importance = []
            sensitivity_analysis_importance = []

            signals_to_analyze = range(0, 3)
            lime_exp = BaselineExplainer(exp.train_loader, exp.valid_loader, exp.test_loader,
                                             exp.feature_size, data_class=exp.patient_data,
                                             data=exp.data, baseline_method='lime')
            for sub_ind, sample_ID in enumerate(samples_to_analyze[exp.data]):
                top_FCC, importance, top_occ, importance_occ, top_occ_aug, importance_occ_aug, top_SA, importance_SA = \
                        exp.plot_baseline(sample_ID, signals_to_analyze, sensitivity_analysis[sub_ind, :, :], data=exp.data, plot=False,
                                          gt_importance_subj=None, tvec=tvec, cv=50)
                all_FFC_importance.append(importance)
                all_AFO_importance.append(importance_occ_aug)
                all_FO_importance.append(importance_occ)

                with open(os.path.join('/scratch/gobi1/sana/TSX_results/simulation_cv_results/results_%scv_0.pkl' %str(sample_ID)), 'rb') as f:
                    arr = pkl.load(f)
                FFC_importance.append(arr['FFC']['imp'])
                AFO_importance.append(arr['AFO']['imp'])
                FO_importance.append(arr['Suresh_et_al']['imp'])
                sensitivity_analysis_importance.append(arr['Sens']['imp'])


            FFC_importance_rndm = np.array(all_FFC_importance)
            AFO_importance_rndm = np.array(all_AFO_importance)
            FO_importance_rndm = np.array(all_FO_importance)
            sensitivity_analysis_rndm = np.array(sensitivity_analysis)
            FFC_importance = np.array(FFC_importance)
            AFO_importance = np.array(AFO_importance)
            FO_importance = np.array(FO_importance)
            sensitivity_analysis_importance = np.array(sensitivity_analysis_importance)


            FFC_importance_rndm = np.nan_to_num(FFC_importance_rndm)
            AFO_importance_rndm = np.nan_to_num(AFO_importance_rndm)
            FO_importance_rndm = np.nan_to_num(FO_importance_rndm)
            sensitivity_analysis_rndm = np.nan_to_num(sensitivity_analysis_rndm)


            FFC_corr = []
            AFO_corr = []
            FO_corr = []
            sens_corr = []
            for sig in range(len(FFC_importance)):
                FFC_corr.append(spearmanr(FFC_importance_rndm[sig].reshape(-1,), FFC_importance[sig].reshape(-1,))[0])
                AFO_corr.append(spearmanr(AFO_importance_rndm[sig].reshape(-1,), AFO_importance[sig].reshape(-1,))[0])
                FO_corr.append(spearmanr(FO_importance_rndm[sig].reshape(-1,), FO_importance[sig].reshape(-1,))[0])
                sens_corr.append(spearmanr(sensitivity_analysis_rndm[sig].reshape(-1,), sensitivity_analysis_importance[sig].reshape(-1,))[0])

            FFC_distance = []
            FO_distance = []
            AFO_distance = []
            sensitivity_distance = []
            for s in range(len(FFC_importance)):
                FFC_distance.append(np.linalg.norm((FFC_importance[s] - FFC_importance_rndm[s])))
                FO_distance.append(np.linalg.norm((FO_importance[s] - FO_importance_rndm[s])))
                AFO_distance.append(np.linalg.norm((AFO_importance[s] - AFO_importance_rndm[s])))
                sensitivity_distance.append(
                    np.linalg.norm((sensitivity_analysis_importance[s] - sensitivity_analysis_rndm[s])))

            print("Printing results for %f percent randomization:"%(ratio*100))
            print("FFC correlation: %.3f +- %.3f" % (np.mean(FFC_corr), np.std(FFC_corr)))
            print("AFO correlation: %.3f +- %.3f" % (np.mean(AFO_corr), np.std(AFO_corr)))
            print("FO correlation: %.3f +- %.3f" % (np.mean(FO_corr), np.std(FO_corr)))
            print("Sensitivity correlation: %.3f +- %.3f" % (np.mean(sens_corr), np.std(sens_corr)))

            print("FFC l2 distance: %.3f +- %.3f" % (np.mean(np.array(FFC_distance)), np.std(np.array(FFC_distance))))
            print("AFO l2 distance: %.3f +- %.3f" % (np.mean(np.array(AFO_distance)), np.std(np.array(AFO_distance))))
            print("FO l2 distance: %.3f +- %.3f" % (np.mean(np.array(FO_distance)), np.std(np.array(FO_distance))))
            print("Sensitivity l2 distance: %.3f +- %.3f" % (
            np.mean(np.array(sensitivity_distance)), np.std(np.array(sensitivity_distance))))
            print('*****************************************************')

            FFC_corr_all.append(FFC_corr)
            AFO_corr_all.append(AFO_corr)
            FO_corr_all.append(FO_corr)
            Sens_corr_all.append(sens_corr)

            with open('/scratch/gobi1/sana/TSX_results/simulation/sanity_checks.pkl', 'wb') as f:
                pkl.dump({'FFC': np.array(FFC_corr_all),
                          'AFO': np.array(AFO_corr_all),
                          'FO': np.array(FO_corr_all),
                          'Sens': np.array(Sens_corr_all)}, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple sanity checks on the data')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--check', type=str, default='randomized_param')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--all_samples', action='store_true')
    args = parser.parse_args()
    np.random.seed(123456)
    if (os.path.exists('/scratch/gobi1/sana/TSX_results/simulation_change_back/sanity_checks.pkl')):
        with open('/scratch/gobi1/sana/TSX_results/simulation_change_back/sanity_checks.pkl', 'rb') as f:
            arr = pkl.load(f)
        y = np.ones(arr['FFC'].shape[0]+1)
        y_std = np.zeros(arr['FFC'].shape[0] + 1)
        y[1:] = np.mean(arr['FFC'], axis=-1)
        y_std[1:] = np.std(arr['FFC'], axis=-1)
        y_sens = np.ones(arr['FFC'].shape[0] + 1)
        y_std_sens = np.zeros(arr['FFC'].shape[0] + 1)
        y_sens[1:] = np.mean(arr['Sens'], axis=-1)
        y_std_sens[1:] = np.std(arr['Sens'], axis=-1)
        x = [0, .2, .4, .6, .8, 1.]
        plt.figure(figsize=(10, 5))
        plt.errorbar(x,y, yerr=y_std, linewidth=3, color='r', marker='o', label='FIT')
        # plt.errorbar(x, y_sens, yerr=y_std_sens, linewidth=3, color='b', marker='o', label='Sensitivity Analysis')
        plt.title('FIT explanation parameter randomization test', fontweight='bold', fontsize=18)
        plt.xlabel('percentage of randomized weights', fontweight='bold', fontsize=14)
        plt.ylabel('Spearman rank correlation', fontweight='bold', fontsize=14)
        plt.legend()
        plt.savefig('/scratch/gobi1/sana/TSX_results/simulation/sanity_checks.pdf', dpi=400)
    else:
        main(train=args.train, data=args.data, all_samples=args.all_samples, check=args.check)
