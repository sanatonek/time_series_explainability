import os
import argparse

import torch
from TSX.utils import load_simulated_data, train_model_rt

from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency
from TSX.models import StateClassifier, RETAIN
from TSX.generator import JointFeatureGenerator
from TSX.explainers import RETAINexplainer, FITExplainer


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100,path='./data/simulated_data', percentage=0.8)

    # Prepare model to explain
    if args.explainer == 'retain':
        model = RETAIN(dim_input=3, dim_emb=64, dropout_emb=0.5, dim_alpha=32, dim_beta=32,
                       dropout_context=0.5, dim_output=2)
        explainer = RETAINexplainer(model, args.data)
        if args.train:
            explainer.fit_model(train_loader, valid_loader, test_loader, plot=True, epochs=100)
        else:
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'retain'))))

    else:
        model = StateClassifier(feature_size=3, n_state=2, hidden_size=100)
        if args.train:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
            train_model_rt(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=50,
                           device=device, experiment='model', data=args.data)
        model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))

        if args.explainer == 'fit':
            explainer = FITExplainer(model, train_loader, valid_loader)
            generator = JointFeatureGenerator(3, hidden_size=10, data=args.data)
            explainer.fit_generator(generator, train_loader, valid_loader)

        elif args.explainer == 'integrated_gradient':
            explainer = IntegratedGradients(model)

        elif args.explainer == 'deep_lift':
            explainer = DeepLift(model)

        elif args.explainer == 'saliency':
            explainer = Saliency(model)

        elif args.explainer == 'gradient_shap':
            explainer = GradientShap(model)

    for x,y in test_loader:
        model.train()
        model.to(device)
        x = x.to(device)
        y = y.to(device)
        if args.explainer == 'gradient_shap':
            rand_img_dist = torch.cat([x[:3] * 0, x[:3] * 1]) #TODO: figure this out!!!
            score = explainer.attribute(x[:3], n_samples=50, stdevs=0.0001,
                                                  baselines=rand_img_dist, target=y[:3, -1].long())
        elif args.explainer == 'deep_lift':
            model.zero_grad()
            score = explainer.attribute(x[:3], target=y[:3, -1].long(), baselines=(x[:3] * 0))
        else:
            score = explainer.attribute(x[:3], y[:3, -1].long())
        print(score.shape)
        break