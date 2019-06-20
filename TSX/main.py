import torch
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from TSX.utils import train_model, load_data, test, load_simulated_data, load_ghg_data
from TSX.models import DeepKnn
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer
import argparse

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

def main(experiment, train, uncertainty_score, sensitivity=False, sim_data=False, data='mimic'):
    print('********** Experiment with the %s model **********' %(experiment))

    # Configurations
    encoding_size = 100
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs=120
    historical=True
    rnn_type='GRU'

    if sim_data:
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, path='./data_generator/data/simulated_data')
        feature_size = p_data.shape[1]
        encoding_size=5
        if experiment =='feature_generator_explainer':
            n_epochs = 50
        elif experiment=='risk_predictor':
            n_epochs = 30
        data='simulation'
        historical=True
    else:
        if data=='mimic':
            p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100,path='./data_generator/data_before')
            feature_size = p_data.feature_size
            encoding_size=150
        elif data=='ghg':
            p_data, train_loader, valid_loader, test_loader = load_ghg_data(batch_size)
            feature_size = p_data.feature_size
            n_epochs=80
            rnn_type='GRU'
            encoding_size=100

    if experiment == 'baseline':
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == 'risk_predictor':
        exp = EncoderPredictor(train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU',simulation=sim_data,data=data)
    elif experiment == 'feature_generator_explainer':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data, generator_hidden_size=encoding_size, prediction_size=1, historical=True, simulation=sim_data,data=data)

    exp.run(train=train,n_epochs=n_epochs)

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
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty, sensitivity=args.sensitivity, sim_data=args.simulation, data=args.data)
