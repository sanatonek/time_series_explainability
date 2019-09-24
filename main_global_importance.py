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

MIMIC_TEST_SAMPLES =  [4387, 481, 546, 10]
#SIMULATION_SAMPLES = list(range(10))
SIMULATION_SAMPLES = []
samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}


def main(experiment, train, uncertainty_score, data, generator_type):
    print('********** Experiment with the %s data **********' %(experiment))
    with open('config.json') as config_file:
        configs = json.load(config_file)[data][experiment]

    if data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=configs['batch_size'],
                                                                    path='./data')
        feature_size = p_data.feature_size
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
        #print(spike_data)
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                        generator_hidden_size=configs['encoding_size'], prediction_size=1, historical=(configs['historical']==1),
                                        generator_type=generator_type, data=data, experiment=experiment+'_'+generator_type,spike_data=spike_data)
    elif experiment == 'lime_explainer':
        exp = BaselineExplainer(train_loader, valid_loader, test_loader, feature_size, data_class=p_data, data=data, baseline_method='lime')

    exp.run(train=train, n_epochs=configs['n_epochs'], samples_to_analyze=samples_to_analyze[data])
    #exp.final_reported_plots(samples_to_analyze=samples_to_analyze[data])

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
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty, data=args.data, generator_type=args.generator)
