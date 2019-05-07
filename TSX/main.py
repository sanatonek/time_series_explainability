import torch
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from TSX.utils import train_model, load_data, test, load_simulated_data
from TSX.models import DeepKnn
from TSX.experiments import KalmanExperiment, Baseline, EncoderPredictor, GeneratorExplainer, FeatureGeneratorExplainer
import argparse
import numpy as np


def main(experiment, train, uncertainty_score, sensitivity=False, sim_data=False):
    print('********** Experiment with the %s model **********' %(experiment))

    # Configurations
    encoding_size = 100
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if sim_data:
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, path='./data_generator/data/simulated_data')
        feature_size = p_data.shape[1]
        encoding_size=20
    else:
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size)
        feature_size = p_data.feature_size

    if experiment == 'baseline':
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == 'risk_predictor':
        exp = EncoderPredictor(train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU',simulation=sim_data)
    elif experiment == 'VAE':
        exp = KalmanExperiment(train_loader, valid_loader, test_loader, p_data.feature_size, encoding_size)
    elif experiment == 'generator_explainer':
        exp = GeneratorExplainer(train_loader, valid_loader, test_loader, p_data.feature_size, encoding_size)
    elif experiment == 'feature_generator_explainer':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, historical=True, simulation=sim_data)

    exp.run(train=train)
    # span = []
    # testset = list(exp.test_loader.dataset)
    # for i,(signal,label) in enumerate(testset):
    #     exp.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
    #     exp.risk_predictor.to(device)
    #     exp.risk_predictor.eval()
    #     risk=[]
    #     for t in range(1,48):
    #         risk.append(exp.risk_predictor(signal[:, 0:t].view(1, signal.shape[0], t).to(device)).item())
    #     span.append((i,max(risk) - min(risk)))
    # span.sort(key= lambda pair:pair[1], reverse=True)
    # print(span[0:100])


    if sensitivity:
        #exp.model.train()
        for i, (signals, labels) in enumerate(train_loader):
            signals = torch.Tensor(signals.float()).to(device).requires_grad_()
            risks = exp.model(signals)
            risks[0].backward(retain_graph=True)
            print(signals.grad.data[0,:,:])


    if uncertainty_score:
        # Evaluate uncertainty using deep KNN method
        print('\n********** Uncertainty Evaluation: **********')
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
    parser.add_argument('--sensitivity',action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty, sensitivity=args.sensitivity, sim_data=args.simulation)
