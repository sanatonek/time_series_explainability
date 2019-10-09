from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.models import DeepKnn
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer, BaselineExplainer

import torch
import os
import sys
import json
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp']

MIMIC_TEST_SAMPLES = [3095, 3022, 1778, 619, 1971, 2702, 1952, 1645, 4552, 1527, 1801]
#[3095, 3022, 1778, 619, 1971, 2702, 1952, 1645, 4552, 1527, 1801, 4006, 2589, 2563, 2110, 1477, 2717, 4387, 223, 2582, 262, 323, 3304, 4449, 345, 2188, 4007, 2076, 80, 4453, 3416, 3985, 8, 1821, 316, 2689, 954, 4010, 283, 3575, 4256, 3045, 673, 4564, 2254, 6, 3120, 4517, 3055, 4402, 2871, 1171, 588, 2242, 3500, 1683, 606, 799, 303, 4155, 2531, 629, 2838, 1534, 343, 1526, 4483, 3586, 2553, 2010, 728, 1966, 950, 170, 2257, 2075, 3882, 3081, 3970, 3127, 1361, 488, 897, 3037, 1680, 647, 986, 3905, 496, 4494, 372, 2644, 764, 3589, 1620, 712, 1717, 3484, 3661, 1235, 1422, 2080, 3305, 2040, 168, 3087, 2335, 687, 399, 4560, 1293, 234, 1030, 4046, 3713, 4420, 2780, 3569, 1994, 4526, 2690, 3057, 2710, 144, 4571, 1671, 3937, 924, 3437, 132, 2171, 210, 341, 4025, 4221, 2475, 3824, 64, 988, 1138, 3155, 2015, 928, 1411, 3029, 4118, 1672, 1826, 1624, 3828, 2952, 2805, 4020, 2818, 993, 3663, 4236, 4078, 3983, 1057, 3843, 4450, 2933, 1718, 3968, 4199, 1835, 2964, 3435, 1769, 1943, 3585, 1317, 3913, 1344, 2208, 122, 3077, 3086, 502, 1381, 2733, 2571, 2085, 586, 3054, 2604, 4258, 946, 2599, 4442, 4417, 956, 2238, 3355, 61, 1052, 930, 169, 3960, 798, 1674, 3445, 523, 3623, 1575, 816, 3509, 1572, 2804, 1279, 82, 2849, 3462, 985, 3324, 1338, 870, 3800, 3056, 2752, 1813, 1987, 193, 1310, 2031, 4370, 3513, 2777, 531, 2745, 2929, 4400, 711, 2945, 3734, 3088, 723, 3313, 4087, 1410, 3013, 3952, 4423, 2299, 2236, 2363, 4326, 4259, 2721, 1239, 2997, 1430, 4467, 2998, 1484, 3896, 2383, 4081, 1176, 4336, 3493, 2910, 260, 4011, 1993, 1863, 1660, 3617, 1563, 784, 4310, 1911, 3096, 682, 3552, 3031, 2621, 2739, 3319, 1428, 1440, 1255, 3873, 2624, 846, 2699, 3603, 2902, 479, 4354, 3168, 3218, 3004, 3784, 4385, 2439, 2260, 624, 3752]

SIMULATION_SAMPLES = [101, 48]#, 88, 192, 143, 166, 18, 58, 172, 132]
#SIMULATION_SAMPLES = []
samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}


def main(experiment, train, uncertainty_score, data, generator_type, all_samples):
    print('********** Experiment with the %s data **********' %(experiment))
    with open('config.json') as config_file:
        configs = json.load(config_file)[data][experiment]

    if data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=configs['batch_size'],
                                                                    path='./data')
        feature_size = p_data.feature_size
        #samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}
    elif data == 'ghg':
        p_data, train_loader, valid_loader, test_loader = load_ghg_data(configs['batch_size'])
        feature_size = p_data.feature_size
    elif data == 'simulation_spike':
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data_generator/data/simulated_data',data_type='spike')
        feature_size = p_data.shape[1]

    elif data == 'simulation':
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data/simulated_data')
        feature_size = p_data.shape[1]

    if data=='simulation_spike':
        data='simulation'
        spike_data=True
    else:
        spike_data=False

    if experiment == 'baseline':
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == 'risk_predictor':
        exp = EncoderPredictor(train_loader, valid_loader, test_loader, feature_size, configs['encoding_size'], rnn_type=configs['rnn_type'], data=data)
    elif experiment == 'feature_generator_explainer':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                        generator_hidden_size=configs['encoding_size'], prediction_size=1, historical=(configs['historical']==1),
                                        generator_type=generator_type, data=data, experiment=experiment+'_'+generator_type,spike_data=spike_data)
    elif experiment == 'lime_explainer':
        exp = BaselineExplainer(train_loader, valid_loader, test_loader, feature_size, data_class=p_data, data=data, baseline_method='lime')

    if data=='mimic' and all_samples:
        print('Running mimic experiment on all test data')
        print('Number of test samples: ', len(exp.test_loader.dataset))
        for i in range(670,len(exp.test_loader.dataset),5):
            exp.run(train=train, n_epochs=configs['n_epochs'], samples_to_analyze=[i,i+1,i+2,i+3,i+4])
    # else:
    #     exp.run(train=train, n_epochs=configs['n_epochs'], samples_to_analyze=samples_to_analyze[data])

    # span = []
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # # import matplotlib.pyplot as plt
    # testset = list(exp.test_loader.dataset)
    # # # signals = torch.stack(([x[0] for x in testset]))
    # # # plt.plot(np.array(signals[4126,2,:]))
    # # # plt.show()
    # for i,(signal,label) in enumerate(testset):
    #     # if i==79:
    #     #     for j in range(31):
    #     #         plt.plot(signal[j,:].cpu().detach().numpy())
    #     #     plt.show()
    #     exp.risk_predictor.load_state_dict(torch.load('./ckpt/mimic/risk_predictor.pt'))
    #     exp.risk_predictor.to(device)
    #     exp.risk_predictor.eval()
    #     risk=[]
    #     for t in range(1,48):
    #         risk.append(exp.risk_predictor(signal[:, 0:t].view(1, signal.shape[0], t).to(device)).item())
    #     span.append((i,max(risk) - min(risk)))
    # span.sort(key= lambda pair:pair[1], reverse=True)
    # print([x[0] for x in span[0:300]])


    
    # if experiment=='feature_generator_explainer':
    #     exp.final_reported_plots(samples_to_analyze=samples_to_analyze[data])

    # For MIMIC experiment, extract population level importance for interventions
    # print('********** Extracting population level intervention statistics **********')
    # if data == 'mimic' and experiment == 'feature_generator_explainer':
    #     for id in range(len(intervention_list)):
    #         if not os.path.exists("./interventions/int_%d.pkl" % (id)):
    #             exp.summary_stat(id)
    #         exp.plot_summary_stat(id)

    if uncertainty_score:
        # Evaluate output uncertainty using deep KNN method
        print('\n********** Uncertainty Evaluation: **********')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sample_ind = 1
        n_nearest_neighbors = 10
        dknn = DeepKnn(exp.model, p_data.train_data[0:int(0.8 * p_data.n_train), :, :],
                    p_data.train_label[0:int(0.8 * p_data.n_train)], device)
        knn_labels = dknn.evaluate_confidence(sample=p_data.test_data[sample_ind, :, :].reshape((1, -1, 48)),
                                              sample_label=p_data.test_label[sample_ind],
                                              _nearest_neighbors=n_nearest_neighbors, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--model', type=str, default='feature_generator_explainer', help='Prediction model')
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--generator', type=str, default='RNN_generator')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--all_samples', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty, data=args.data, generator_type=args.generator, all_samples=args.all_samples)
