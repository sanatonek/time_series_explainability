import torch
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from TSX.utils import load_data, load_simulated_data
from TSX.models import DeepKnn
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer
import argparse

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']


def main(experiment, train, uncertainty_score, sim_data=False):
    print('********** Experiment with the %s model **********' %(experiment))

    if sim_data:
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, path='./data_generator/data/simulated_data')
        feature_size = p_data.shape[1]
        encoding_size=20
    else:
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path='./data_generator/data_before')
        feature_size = p_data.feature_size
        encoding_size = 150

    if experiment == 'baseline':
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == 'risk_predictor':
        exp = EncoderPredictor(train_loader, valid_loader, test_loader, feature_size, encoding_size=encoding_size, rnn_type='GRU',simulation=sim_data)
    elif experiment == 'feature_generator_explainer':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data= p_data,
                                        generator_hidden_size=80, prediction_size=1, historical=True, simulation=sim_data)

    exp.run(train=train)

    # For MIMIC experiment, extract population level importance for interventions
    if not sim_data:
        for id in range(len(intervention_list)):
            exp.summary_stat(id)
            exp.plot_summary_stat(id)

    if uncertainty_score:
        # Evaluate uncertainty using deep KNN method
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
    parser.add_argument('--simulation', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty, sim_data=args.simulation)
