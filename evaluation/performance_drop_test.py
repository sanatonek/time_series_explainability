import os
import pickle as pkl
import numpy as np
import argparse

from sklearn import metrics
import torch

from TSX.models import StateClassifier

def main(args):
    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
    elif args.data == 'simulation_l2x':
        feature_size = 3
        data_path = './data/simulated_data_l2x'
    output_path = '/scratch/gobi1/sana/TSX_results/new_results/%s' % args.data

    with open(os.path.join(data_path, 'state_dataset_x_test.pkl'), 'rb') as f:
        x_test = pkl.load(f)
    with open(os.path.join(data_path, 'state_dataset_y_test.pkl'), 'rb') as f:
        y_test = pkl.load(f)
    with open(os.path.join(output_path, '%s_test_importance_scores.pkl' %args.explainer), 'rb') as f:
        importance_scores = pkl.load(f)


    model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=200)
    model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))
    model.eval()

    min_t = 30

    y1, y2, label = [], [], []
    for i,x in enumerate(x_test):
        imp = np.unravel_index(importance_scores[i,:,min_t:].argmax(), importance_scores[i,:,min_t:].shape)
        sample = x[:, :imp[1]+min_t+1]
        label.append(y_test[i,imp[1]+min_t])
        y = torch.nn.Softmax(-1)(model(torch.Tensor(sample).unsqueeze(0)))[:, 1]
        y1.append(y.detach().cpu().numpy())
        x_cf = sample.copy()
        x_cf[imp[0],-1] = x_cf[imp[0],-2]
        y = torch.nn.Softmax(-1)(model(torch.Tensor(x_cf).unsqueeze(0)))[:, 1]
        y2.append(y.detach().cpu().numpy())

    original_auc = metrics.roc_auc_score(np.array(label), np.array(y1))
    modified_auc = metrics.roc_auc_score(np.array(label), np.array(y2))
    print(original_auc, modified_auc)


if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run AUC drop test')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--generator_type', type=str, default='history')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)