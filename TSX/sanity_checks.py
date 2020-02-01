from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer, BaselineExplainer

import torch
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate as correlate
from scipy.stats import spearmanr
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
                      #  3585, 1484, 687, 3783, 488, 3661, 2312, 4453, 1239, 3575, 4258, 4501, 1993, 2257, 4020, 673, 945,
                      #  4430, 1835, 3558, 2331, 1361, 1769, 2176, 1826, 4494, 2330, 2689, 950, 2188, 1279, 4078, 283,
                      #  3029, 4127, 4560, 2414, 3049, 4387, 210, 4195, 2208, 2015, 3319, 2752, 4474, 372, 3493, 4366,
                      #  169, 2363, 2441, 4025, 1293, 1674, 4259, 168, 2260, 2868, 3882, 2040, 234, 988, 61, 3082, 1813,
                      #  491, 4118, 3985, 2805, 723, 3355, 3752, 3077, 496, 629, 4155, 1289, 1943, 1187, 1572, 998, 303,
                      #  682, 3968, 2080, 4526, 2383, 3088, 3937, 2010, 2780, 3081, 2628, 3776, 3275, 3057, 1987, 134,
                      #  2571, 3013, 4423, 4402, 2591, 4161, 2588, 2470, 2818, 1333, 64, 4446, 3828, 3168, 1911, 1704,
                      #  2171, 2335, 4105, 2085, 720, 728, 3218, 2032, 3513, 2491, 1381, 2910, 2075, 3462, 3644, 4469,
                      #  2319, 2242, 1054, 1440, 1961, 1344, 1276, 798, 4221, 2501, 2170, 2933, 2031, 1717, 4417, 799,
                      #  2284, 985, 4087, 1895, 3552, 902, 2475, 1355, 3809, 2299, 3604, 4236, 4256, 1966, 3313, 1863,
                      #  2934, 4370, 3132, 3983, 711, 227, 2346, 2464, 2246, 4046, 4541, 170, 2410, 4211, 36, 986, 4442,
                      # 712, 60, 2918, 3646, 1152, 4055, 3190, 3505, 3165, 977, 4058, 3913, 502, 2264, 3031, 2237, 4326,
                      # 1623, 144, 4171, 2029]

SIMULATION_SAMPLES = np.random.choice(100, 80)#[101, 48]#, 88]#, 192, 143, 166, 18, 58, 172, 132]
samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}


def main(train, data, all_samples, check):
    experiment = "saliency test"
    batch_size = 100
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

    if check=='randomized_param' or check=='randomized_data':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                            generator_hidden_size=configs['encoding_size'], prediction_size=1,
                                            historical=(configs['historical'] == 1),
                                            generator_type=generator_type, data=data,
                                            experiment=experiment + '_' + generator_type, spike_data=spike_data)

        if all_samples:
            print('Experiment on all test data for %s'%check)
            print('Number of test samples: ', len(exp.test_loader.dataset))
            exp.run(train=train, n_epochs=configs['n_epochs'], samples_to_analyze=range(0,len(exp.test_loader.dataset)),
                    sanity_check=check, plot=False)
        else:
            FFC_importance_rndm, AFO_importance_rndm, FO_importance_rndm, lime_importance_rndm, sensitivity_analysis_rndm = exp.run(
                train=train, n_epochs=configs['n_epochs'], samples_to_analyze=samples_to_analyze[data], plot=False, sanity_check=check)
            FFC_importance, AFO_importance, FO_importance, lime_importance, sensitivity_analysis = exp.run(
                train=train, n_epochs=configs['n_epochs'], samples_to_analyze=samples_to_analyze[data], plot=False)

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
                sens_corr.append(spearmanr(sensitivity_analysis_rndm[sig].reshape(-1,), sensitivity_analysis[sig].reshape(-1,))[0])

            # FFC_corr, _ = pearson_corr(FFC_importance_rndm.reshape(-1,), FFC_importance.reshape(-1,))
            # AFO_corr, _ = pearson_corr(AFO_importance_rndm.reshape(-1,), AFO_importance.reshape(-1,))
            # FO_corr, _  = pearson_corr(FO_importance_rndm.reshape(-1,), FO_importance.reshape(-1,))
            # # lime_corr = pearson_corr(lime_importance_rndm.reshape(-1,), lime_importance.reshape(-1,))
            # sens_corr, _ = pearson_corr(sensitivity_analysis_rndm.reshape(-1,), sensitivity_analysis.reshape(-1,))
            with open('./output/%s_%s_check_correlations.txt'%(data, check), 'w') as f:
                f.write('FFC: %f'%np.mean(FFC_corr))
                f.write('\nAFO: %f'%np.mean(AFO_corr))
                f.write('\nFO: %f'%np.mean(FO_corr))
                f.write('\nSensitivity analysis: %f'%np.mean(sens_corr))

    elif check == 'randomized_model':
        exp_RNN = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                            generator_hidden_size=configs['encoding_size'], prediction_size=1,
                                            historical=(configs['historical'] == 1),
                                            generator_type=generator_type, data=data,
                                            experiment=experiment + '_' + generator_type, spike_data=spike_data)
        exp_LR = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                            generator_hidden_size=configs['encoding_size'], prediction_size=1,
                                            historical=(configs['historical'] == 1),
                                            generator_type=generator_type, data=data,
                                            experiment=experiment + '_' + generator_type, spike_data=spike_data)
        if all_samples:
            print('Experiment on all test data for %s'%check)
            print('Number of test samples: ', len(exp_LR.test_loader.dataset))
            exp_LR.run(train=train, n_epochs=configs['n_epochs'], samples_to_analyze=range(0,len(exp_LR.test_loader.dataset)),
                    sanity_check=check, plot=False)
        else:
            FFC_importance_rndm, AFO_importance_rndm, FO_importance_rndm, lime_importance_rndm, sensitivity_analysis_rndm = exp_LR.run(
                train=train, n_epochs=configs['n_epochs'], samples_to_analyze=samples_to_analyze[data], sanity_check="randomized_model")
            FFC_importance, AFO_importance, FO_importance, lime_importance, sensitivity_analysis = exp_RNN.run(
                train=train, n_epochs=configs['n_epochs'], samples_to_analyze=samples_to_analyze[data])

    if not all_samples:
        FFC_distance = []
        FO_distance = []
        AFO_distance = []
        sensitivity_distance = []
        for s in range(len(FFC_importance)):
            FFC_distance.append(np.linalg.norm((FFC_importance[s]-FFC_importance_rndm[s])))
            FO_distance.append(np.linalg.norm((FO_importance[s] - FO_importance_rndm[s])))
            AFO_distance.append(np.linalg.norm((AFO_importance[s] - AFO_importance_rndm[s])))
            sensitivity_distance.append(np.linalg.norm((sensitivity_analysis[s] - sensitivity_analysis_rndm[s])))
        print("FFC correlation: %.3f +- %.3f" % (np.mean(FFC_corr), np.std(FFC_corr)))
        print("AFO correlation: %.3f +- %.3f" % (np.mean(AFO_corr), np.std(AFO_corr)))
        print("FO correlation: %.3f +- %.3f" % (np.mean(FO_corr), np.std(FO_corr)))
        print("Sensitivity correlation: %.3f +- %.3f" % (np.mean(sens_corr), np.std(sens_corr)))

        print("FFC l2 distance: %.3f +- %.3f"%(np.mean(np.array(FFC_distance)),np.std(np.array(FFC_distance))) )
        print("AFO l2 distance: %.3f +- %.3f"%(np.mean(np.array(AFO_distance)),np.std(np.array(AFO_distance))))
        print("FO l2 distance: %.3f +- %.3f"%(np.mean(np.array(FO_distance)),np.std(np.array(FO_distance))))
        print("Sensitivity l2 distance: %.3f +- %.3f"%(np.mean(np.array(sensitivity_distance)),np.std(np.array(sensitivity_distance))))

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        # ax1.plot(FFC_importance[0,2,:], c='g', label='trained model')
        # ax1.plot(FFC_importance_rndm[0, 2, :], c='r', label=check)
        # ax1.set_title("FFC")
        # ax2.plot(AFO_importance[0,2,:], c='g', label='trained model')
        # ax2.plot(AFO_importance_rndm[0, 2, :], c='r', label=check)
        # ax2.set_title("AFO")
        # ax3.plot(FO_importance[0,2,:], c='g', label='trained model')
        # ax3.plot(FO_importance_rndm[0, 2, :], c='r', label=check)
        # ax3.set_title("FO")
        # ax4.plot(sensitivity_analysis[0,2,:], c='g', label='trained model')
        # ax4.plot(sensitivity_analysis_rndm[0, 2, :], c='r', label=check)
        # ax4.set_title("Sensitivity analysis")
        #
        # ax1.legend()
        # f.set_figheight(15)
        # f.set_figwidth(20)
        # plt.savefig(os.path.join('./examples',data,'sanity_check_%s.pdf'%check), dpi=300, orientation='landscape')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple sanity checks on the data')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--check', type=str, default='randomized_param')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--all_samples', action='store_true')
    args = parser.parse_args()
    np.random.seed(123456)
    main(train=args.train, data=args.data, all_samples=args.all_samples, check=args.check)
