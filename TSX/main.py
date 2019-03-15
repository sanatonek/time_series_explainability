import torch
import os
from TSX.utils import train_model, load_data, test
from TSX.models import EncoderRNN, RiskPredictor, LR, DeepKnn
import argparse


def main(experiment, train):
    print('********** Experiment with the %s model **********' %(experiment))

    # Configurations
    encoding_size = 100
    batch_size = 100
    n_epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p_data, train_loader, valid_loader, test_loader = load_data(batch_size)

    if experiment == 'LR':
        model = LR(feature_size=p_data.feature_size)
    elif experiment == 'encoder':
        model = EncoderRNN(feature_size=p_data.feature_size, hidden_size=encoding_size, rnn='GRU')
    elif experiment == 'risk_predictor':
        state_encoder = EncoderRNN(feature_size=p_data.feature_size, hidden_size=encoding_size, rnn='GRU', regres=False)
        risk_predictor = RiskPredictor(encoding_size=encoding_size)
        model = torch.nn.Sequential(state_encoder, risk_predictor)
    model = model.to(device)

    if train:
        train_model(model, train_loader, valid_loader, n_epochs, device, experiment)
        # Evaluate performance on held-out test set
        _, _, auc_test, correct_label, test_loss = test(test_loader, model, device)
        print('\nFinal performance on held out test set ===> AUC: ', auc_test)

    else:
        if os.path.exists('./ckpt/' + str(experiment) + '.pt'):
            model.load_state_dict(torch.load('./ckpt/' + str(experiment) + '.pt'))
        else:
            raise RuntimeError('No saved checkpoint for this model')

    # Evaluate uncertainty using deep KNN method
    print('\n********** Uncertainty Evaluation: **********')
    sample_ind = 1
    n_nearest_neighbors = 10
    dknn = DeepKnn(model, p_data.train_data[0:int(0.8 * p_data.n_train), :, :],
                    p_data.train_label[0:int(0.8 * p_data.n_train)], device)
    knn_labels = dknn.evaluate_confidence(sample=p_data.test_data[sample_ind, :, :].reshape((1, -1, 48)),
                                          sample_label=p_data.test_label[sample_ind],
                                          n_nearest_neighbors=n_nearest_neighbors, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--model', type=str, default='encoder', help='Prediction model')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train)
