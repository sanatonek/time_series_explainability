from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.experiments import FeatureGeneratorExplainer

import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus',
                     'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp']

MIMIC_TEST_SAMPLES = [3095, 1971, 1778, 1477, 3022, 8, 262, 3437, 1534, 619, 2076, 1801, 4006, 6, 1952, 2582, 4552]

SIMULATION_SAMPLES = [51, 11, 101, 48]
samples_to_analyze = {'mimic': MIMIC_TEST_SAMPLES, 'simulation': SIMULATION_SAMPLES, 'ghg': [], 'simulation_spike': []}


def main(data, generator_type, all_samples, cv=0):
    print('********** Experiment with the %s data **********' % ("feature_generator_explainer"))
    with open('config.json') as config_file:
        configs = json.load(config_file)[data]["feature_generator_explainer"]

    if data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=configs['batch_size'],
                                                                    path='./data', cv=cv)
        feature_size = p_data.feature_size
        # samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}
    elif data == 'ghg':
        p_data, train_loader, valid_loader, test_loader = load_ghg_data(configs['batch_size'], cv=cv)
        feature_size = p_data.feature_size
    elif data == 'simulation_spike':
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data/simulated_spike_data',
                                                                              data_type='spike', cv=cv)
        feature_size = p_data.shape[1]

    elif data == 'simulation':
        percentage = 100.
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data/simulated_data',
                                                                              percentage=percentage / 100, cv=cv)
        # generator_type = generator_type+'_%d'%percentage
        feature_size = p_data.shape[1]

    exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                    generator_hidden_size=configs['encoding_size'], prediction_size=1,
                                    historical=(configs['historical'] == 1),
                                    generator_type=generator_type, data=data,
                                    experiment='feature_generator_explainer_' + generator_type)

    if all_samples:
        print('Experiment on all test data')
        print('Number of test samples: ', len(exp.test_loader.dataset))
        exp.select_top_features(samples_to_analyze = range(0, len(exp.test_loader.dataset) // 2), sub_features=[[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]])
    else:
        exp.select_top_features(samples_to_analyze[data], sub_features=[[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]])
        # print(sub_groups[samples_to_analyze[2]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--generator', type=str, default='joint_RNN_generator')
    parser.add_argument('--predictor', type=str, default='RNN')
    parser.add_argument('--all_samples', action='store_true')
    args = parser.parse_args()
    main(data=args.data, generator_type=args.generator, all_samples=args.all_samples)

