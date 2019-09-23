from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.models import DeepKnn, EncoderRNN
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer, BaselineExplainer
from data_generator.true_generator_state_data import TrueFeatureGenerator
import matplotlib.pyplot as plt

import torch
import os
import sys
import json
import argparse
import numpy as np
import pickle as pkl
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp']

MIMIC_TEST_SAMPLES =  [4387, 481, 546, 10]
SIMULATION_SAMPLES = [101, 48]#, 88, 192, 143, 166, 18, 58, 172, 132]
#SIMULATION_SAMPLES = []
samples_to_analyze = {'mimic':MIMIC_TEST_SAMPLES, 'simulation':SIMULATION_SAMPLES, 'ghg':[], 'simulation_spike':[]}


def main(data, generator_type):
    print('********** Running Generator Baseline Experiment **********')
    with open('config.json') as config_file:
        configs = json.load(config_file)[data]['feature_generator_explainer']

    experiment = 'feature_generator_explainer'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                        generator_hidden_size=configs['encoding_size'], prediction_size=1, historical=(configs['historical']==1), generator_type=generator_type, data=data, experiment=experiment+'_'+generator_type,spike_data=spike_data)

    exp.run(train=False, n_epochs=configs['n_epochs'], samples_to_analyze=samples_to_analyze[data])

    testset = list(exp.test_loader.dataset)
    test_signals = torch.stack(([x[0] for x in testset])).to(device)
    for sample_ID in samples_to_analyze[data]: 
        result_path = '/scratch/gobi1/sana/TSX_results/'+str(data)+'/results_%s.pkl'%str(sample_ID)
        print('Reading results from ', result_path)
        with open(result_path, 'rb') as f:
            arr = pkl.load(f)

        test_signal = test_signals[sample_ID,:,:]
        fcc_importance = arr['FFC']['imp']   
        afo_importance = arr['AFO']['imp']   
        fo_importance = arr['Suresh_et_al']['imp']
        baseline_importance = find_true_gen_importance(test_signal, data)

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        for ft in range(test_signal.shape[0]):
            ax2.plot(fcc_importance[ft,:], linestyle='--', marker='o', markersize=10, label='feature %d'%ft)
            ax2.set_title('FFC', fontweight='bold', fontsize=34)
            ax3.plot(afo_importance[ft,:], linestyle='--', marker='o', markersize=10, label='feature %d'%ft)
            ax3.set_title('AFO', fontweight='bold', fontsize=34)
            ax4.plot(fo_importance[ft,:], linestyle='--', marker='o', markersize=10, label='feature %d'%ft)
            ax4.set_title('Suresh et al', fontweight='bold', fontsize=34)
            ax1.plot(baseline_importance[ft,:], linestyle='--', marker='o', markersize=10, label='feature %d'%ft)
            ax1.set_title('True Generator', fontweight='bold', fontsize=34)
            ax1.set_ylabel('importance score', fontweight='bold', fontsize=18)
            ax2.set_ylabel('importance score', fontweight='bold', fontsize=18)
            ax3.set_ylabel('importance score', fontweight='bold', fontsize=18)
            ax4.set_ylabel('importance score', fontweight='bold', fontsize=18)
            ax1.tick_params(axis='both', labelsize=26)
            ax2.tick_params(axis='both', labelsize=26)
            ax3.tick_params(axis='both', labelsize=26)
            ax4.tick_params(axis='both', labelsize=26)
            ax4.set_xlabel('time', fontweight='bold', fontsize=24)
        f.set_figheight(25)
        f.set_figwidth(30)
        plt.savefig(os.path.join('/scratch/gobi1/sana/TSX_results',data,'generator_baselines_%d.pdf' %(sample_ID)), dpi=300, orientation='landscape', bbox_inches='tight')

def find_true_gen_importance(sample, data):
    true_generator = TrueFeatureGenerator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('config.json') as config_file:
        predictor_configs = json.load(config_file)[data]['risk_predictor']
    predictor_model = EncoderRNN(sample.shape[0], hidden_size=predictor_configs['encoding_size'], rnn=predictor_configs['rnn_type'], regres=True, data=data)
    
    importance = np.zeros((sample.shape[0], sample.shape[1]-1))
    mean_predicted_risk = []
    predictor_model.load_state_dict(torch.load(os.path.join('./ckpt/',data, 'risk_predictor.pt')))
    predictor_model.to(device)
    predictor_model.eval()
    true_risk = np.zeros((sample.shape[-1],))
    n_feature, signal_len = sample.shape
    for t in range(1,signal_len):
        true_risk[t-1] = predictor_model(sample[:,:t+1].view(1,n_feature,-1)).item()
    for f in range(sample.shape[0]):
        sample.to(device)
        for t in range(1,signal_len):
            counterfact = true_generator.sample(sample.cpu().numpy(), t, f)
            full_counterfact = torch.cat((sample[:f,t], torch.Tensor([counterfact]).to(device), sample[f+1:,t]))
            predicted_risk = predictor_model(torch.cat((sample[:,:t],full_counterfact.view(-1,1)),-1).view(1,n_feature,-1))
            importance[f,t-1] = abs(true_risk[t-1]-predicted_risk.item())
    return(importance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--generator', type=str, default='joint_RNN_generator')
    args = parser.parse_args()
    main(data=args.data, generator_type=args.generator)

