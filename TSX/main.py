import torch
import os
from TSX.utils import train_model, load_data, test
from TSX.models import DeepKnn
from TSX.experiments import KalmanExperiment, Baseline, EncoderPredictor
import argparse


def main(experiment, train, uncertainty_score):
    print('********** Experiment with the %s model **********' %(experiment))

    # Configurations
    encoding_size = 100
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p_data, train_loader, valid_loader, test_loader = load_data(batch_size)

    if experiment == 'baseline':
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == 'risk_predictor':
        exp = EncoderPredictor(train_loader, valid_loader, test_loader, p_data.feature_size, encoding_size, rnn_type='GRU')
    elif experiment == 'VAE':
        exp = KalmanExperiment(train_loader, valid_loader, test_loader, p_data.feature_size, encoding_size)

    exp.run(train=train)

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
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty)
