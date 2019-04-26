import torch
from TSX.utils import train_model, load_data, test, load_simulated_data
from TSX.models import DeepKnn
from TSX.experiments import KalmanExperiment, Baseline, EncoderPredictor, GeneratorExplainer, FeatureGeneratorExplainer
import argparse


def main(experiment, train, uncertainty_score, sensitivity=False, sim_data=False):
    print('********** Experiment with the %s model **********' %(experiment))

    # Configurations
    encoding_size = 100
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if sim_data:
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, path='./data_generator/data/simulated_data')
        feature_size = p_data.shape[1]
    else:
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size)
        feature_size = p_data.feature_size

    if experiment == 'baseline':
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == 'risk_predictor':
        exp = EncoderPredictor(train_loader, valid_loader, test_loader, p_data.feature_size, encoding_size, rnn_type='GRU')
    elif experiment == 'VAE':
        exp = KalmanExperiment(train_loader, valid_loader, test_loader, p_data.feature_size, encoding_size)
    elif experiment == 'generator_explainer':
        exp = GeneratorExplainer(train_loader, valid_loader, test_loader, p_data.feature_size, encoding_size)
    elif experiment == 'feature_generator_explainer':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, historical=True, simulation=sim_data)

    exp.run(train=train)

    if sensitivity:
        exp.model.train()
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
    parser.add_argument('--model', type=str, default='encoder', help='Prediction model')
    parser.add_argument('--simulation', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty, sim_data=args.simulation)
