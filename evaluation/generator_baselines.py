from TSX.utils import load_data, load_simulated_data, load_ghg_data
from TSX.models import EncoderRNN
from TSX.experiments import FeatureGeneratorExplainer
from data_generator.true_generator_state_data import TrueFeatureGenerator

import torch
import os
import sys
import json
import argparse
import numpy as np
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def main(data, generator_type, output_path, predictor_model):
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

    testset = list(exp.test_loader.dataset)
    test_signals = torch.stack(([x[0] for x in testset])).to(device)

    true_generator = TrueFeatureGenerator()

    S = 100
    for s in range(S):
        print('generating sample: ', s)
        signal = test_signals[s]
        ffc_sample = np.zeros((test_signals.shape[1], test_signals.shape[-1] * S))
        true_sample = np.zeros((test_signals.shape[1], test_signals.shape[-1] * S))
        for t in range(1, test_signals.shape[-1], 3):
            if t % 3 == 0:
                print('t: ', t)
            ffc_sample_t = exp.generator.forward_joint(signal[:, 0:t].unsqueeze(0))
            ffc_sample[:, s * test_signals.shape[-1] + t] = ffc_sample_t.cpu().detach().numpy()[0]
            true_sample[:, s * test_signals.shape[-1] + t] = true_generator.sample(signal[:, 0:t], t)

    for f in range(test_signals.shape[1]):
        ks_stat_f, p_value = stats.ks_2samp(ffc_sample[f, :], true_sample[f, :])
        print('feature: ', f, 'KS_stat: ', ks_stat_f, 'p_value: ', p_value)


def find_true_gen_importance(sample, data):
    true_generator = TrueFeatureGenerator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('config.json') as config_file:
        predictor_configs = json.load(config_file)[data]['risk_predictor']
    predictor_model = EncoderRNN(sample.shape[0], hidden_size=predictor_configs['encoding_size'],
                                 rnn=predictor_configs['rnn_type'], regres=True, data=data)

    importance = np.zeros((sample.shape[0], sample.shape[1] - 1))
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

    return (importance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--generator', type=str, default='joint_RNN_generator')
    parser.add_argument('--predictor', type=str, default='RNN')
    parser.add_argument('--out', type=str, default='./out')
    args = parser.parse_args()
    if args.out == './out' and not os.path.exists('./out'):
        os.mkdir('./out')
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    main(data=args.data, generator_type=args.generator, output_path=args.out, predictor_model=args.predictor)
