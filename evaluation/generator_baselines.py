from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.models import DeepKnn, EncoderRNN
from TSX.experiments import EncoderPredictor, FeatureGeneratorExplainer, BaselineExplainer
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
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus',
                     'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp']

MIMIC_TEST_SAMPLES = [4387, 481, 546, 10]
SIMULATION_SAMPLES = [101, 48]  # , 88, 192, 143, 166, 18, 58, 172, 132]
# SIMULATION_SAMPLES = []
samples_to_analyze = {'mimic': MIMIC_TEST_SAMPLES, 'simulation': SIMULATION_SAMPLES, 'ghg': [], 'simulation_spike': []}


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
                                                                              path='./data_generator/data/simulated_data',
                                                                              data_type='spike')
        feature_size = p_data.shape[1]

    elif data == 'simulation':
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=configs['batch_size'],
                                                                              path='./data/simulated_data')
        feature_size = p_data.shape[1]

    if data == 'simulation_spike':
        data = 'simulation'
        spike_data = True
    else:
        spike_data = False

    exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data,
                                    generator_hidden_size=configs['encoding_size'], prediction_size=1,
                                    historical=(configs['historical'] == 1), generator_type=generator_type, data=data,
                                    experiment=experiment + '_' + generator_type, spike_data=spike_data)

    exp.run(train=False, n_epochs=configs['n_epochs'], samples_to_analyze=[1,2])

    testset = list(exp.test_loader.dataset)
    test_signals = torch.stack(([x[0] for x in testset])).to(device)

    true_generator = TrueFeatureGenerator()
    ffc_mse = 0
    afo_mse = 0
    fo_mse = 0
    n = 200*test_signals.shape[1]*(test_signals.shape[-1]-1)
    
    ffc_log_prob=0
    afo_log_prob=0
    fo_log_prob=0
    for s in range(200):
        signal = test_signals[s]
        for t in range(1, test_signals.shape[-1]):
            ffc_sample = np.zeros((1,test_signals.shape[1]))
            afo_sample = np.zeros((1,test_signals.shape[1]))
            fo_sample = np.zeros((1,test_signals.shape[1]))
            for f in range(test_signals.shape[1]):
                feature_dist = (np.array(exp.feature_dist[:, f, :]).reshape(-1))
                # feature_dist_1 = (np.array(exp.feature_dist_1[:, f, :]).reshape(-1))
                ffc_sample_t, _ = exp.generator(signal[: , t], signal[:, 0:t].view(1, signal.size(0), t), f, method='m1')
                #print(ffc_sample_t.cpu().detach().numpy())
                ffc_sample[0,f] = ffc_sample_t.cpu().detach().numpy()[0][0]
                #fo_sample[0,f] = torch.Tensor(np.array([np.random.uniform(-3, 3)]).reshape(-1)).detach().numpy()[0]
                fo_sample[0,f] = np.random.uniform(-3, 3,size=1)
                afo_sample[0,f] = np.random.choice(feature_dist,size=1)
                # if exp.risk_predictor(signal[:, 0:t].view(1, signal.shape[0], t)).item() > 0.5:
                #     afo_sample = torch.Tensor(np.array(np.random.choice(feature_dist_0)).reshape(-1, )).detach().numpy()
                # else:
                #     afo_sample = torch.Tensor(np.array(np.random.choice(feature_dist_1)).reshape(-1, )).detach().numpy()
                print(ffc_sample.shape)
            ffc_log_prob += true_generator.log_prob(signal.cpu().numpy(), t, f,ffc_sample)
            afo_log_prob += true_generator.log_prob(signal.cpu().numpy(), t, f,afo_sample)
            fo_log_prob += true_generator.log_prob(signal.cpu().numpy(), t, f,fo_sample)
        # true_sample = np.float(signal[f,t])
                #ffc_mse += np.mean(np.square(ffc_sample-true_sample))
                #afo_mse += np.mean(np.square(afo_sample-true_sample))
                #fo_mse = fo_mse + np.mean(np.square(fo_sample - true_sample))
    print(ffc_log_prob, afo_log_prob, fo_log_prob)
    


    '''
    for sample_ID in samples_to_analyze[data]:
        # result_path = '/scratch/gobi1/sana/TSX_results/' + str(data) + '/results_%s.pkl' % str(sample_ID)
        result_path = '/scratch/gobi1/shalmali/' + str(data) + '/results_%s.pkl' % str(sample_ID)
        with open(result_path, 'rb') as f:
            arr = pkl.load(f)

        test_signal = test_signals[sample_ID, :, :]
        fcc_importance = arr['FFC']['imp']
        afo_importance = arr['AFO']['imp']
        fo_importance = arr['Suresh_et_al']['imp']

        baseline_importance = find_true_gen_importance(test_signal, data,sample_ID)
        result_path = '/scratch/gobi1/shalmali/' + str(data) + '/results_true_%s.pkl' % str(sample_ID)
        #with open(result_path, 'rb') as f:
        #    baseline_importance = pkl.load(f)

        # ffc_mse = ffc_mse + np.mean(np.square(fcc_importance-baseline_importance))
        # afo_mse = afo_mse + np.mean(np.square(afo_importance-baseline_importance))
        # fo_mse = fo_mse + np.mean(np.square(fo_importance - baseline_importance))
        # print(ffc_mse, afo_mse, fo_mse)

    # print('FINAL', ffc_mse, afo_mse, fo_mse)

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        for ft in range(test_signal.shape[0]):
            ax2.plot(fcc_importance[ft, :], linestyle='--', marker='o', markersize=10, label='feature %d' % ft)
            ax2.set_title('FFC', fontweight='bold', fontsize=38)
            ax3.plot(afo_importance[ft, :], linestyle='--', marker='o', markersize=10, label='feature %d' % ft)
            ax3.set_title('AFO', fontweight='bold', fontsize=38)
            ax4.plot(fo_importance[ft, :], linestyle='--', marker='o', markersize=10, label='feature %d' % ft)
            ax4.set_title('FO', fontweight='bold', fontsize=38)
            ax1.plot(baseline_importance[ft, :], linestyle='--', marker='o', markersize=10, label='feature %d' % ft)
            ax1.set_title('True Generator', fontweight='bold', fontsize=34)
            ax1.set_ylabel('importance score', fontweight='bold', fontsize=28)
            ax2.set_ylabel('importance score', fontweight='bold', fontsize=28)
            ax3.set_ylabel('importance score', fontweight='bold', fontsize=28)
            ax4.set_ylabel('importance score', fontweight='bold', fontsize=28)
            ax1.tick_params(axis='both', labelsize=32)
            ax2.tick_params(axis='both', labelsize=32)
            ax3.tick_params(axis='both', labelsize=32)
            ax4.tick_params(axis='both', labelsize=32)
            ax4.set_xlabel('time', fontweight='bold', fontsize=32)
        f.set_figheight(20)
        f.set_figwidth(30)
        plt.savefig(os.path.join('/scratch/gobi1/sana/TSX_results', data, 'generator_baselines_%d.pdf' % (sample_ID)),
                    dpi=300, orientation='landscape', bbox_inches='tight')
    '''

def find_true_gen_importance(sample, data, sampleID):
    true_generator = TrueFeatureGenerator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('config.json') as config_file:
        predictor_configs = json.load(config_file)[data]['risk_predictor']
    predictor_model = EncoderRNN(sample.shape[0], hidden_size=predictor_configs['encoding_size'],
                                 rnn=predictor_configs['rnn_type'], regres=True, data=data)

    importance = np.zeros((sample.shape[0], sample.shape[1] - 1))
    mean_predicted_risk = []
    predictor_model.load_state_dict(torch.load(os.path.join('./ckpt/', data, 'risk_predictor.pt')))
    predictor_model.to(device)
    predictor_model.eval()
    true_risk = np.zeros((sample.shape[-1],))
    n_feature, signal_len = sample.shape
    for t in range(1, signal_len):
        true_risk[t - 1] = predictor_model(sample[:, :t + 1].view(1, n_feature, -1)).item()
    for f in range(sample.shape[0]):
        sample.to(device)
        for t in range(1, signal_len):
            counterfact = true_generator.sample(sample.cpu().numpy(), t, f)
            full_counterfact = torch.cat((sample[:f, t], torch.Tensor([counterfact]).to(device), sample[f + 1:, t]))
            predicted_risk = predictor_model(
                torch.cat((sample[:, :t], full_counterfact.view(-1, 1)), -1).view(1, n_feature, -1))
            importance[f, t - 1] = abs(true_risk[t - 1] - predicted_risk.item())

   # with open(os.path.join('/scratch/gobi1/shalmali', data, '/results_' + str(sampleID) + '.pkl'), 'wb') as f:
    #    pkl.dump(importance)

    return (importance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--generator', type=str, default='joint_RNN_generator')
    args = parser.parse_args()
    main(data=args.data, generator_type=args.generator)
