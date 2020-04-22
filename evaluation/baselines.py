import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import time

from TSX.utils import load_simulated_data, train_model_rt, shade_state
from TSX.models import StateClassifier, RETAIN
from TSX.generator import JointFeatureGenerator
from TSX.explainers import RETAINexplainer, FITExplainer, IGExplainer, FFCExplainer, \
    DeepLiftExplainer, GradientShapExplainer, AFOExplainer, FOExplainer
from sklearn import metrics

# from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency

if __name__=='__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.data=='simulation':
        feature_size = 3
        data_path = './data/simulated_data'
    elif args.data == 'simulation_l2x':
        feature_size = 10
        data_path = './data/simulated_data_l2x'
    output_path = '/scratch/gobi1/shalmali/TSX_results/new_results/%s'%args.data
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_path = os.path.join('./plots/%s' % args.data)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # Load data
    _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, path=data_path, percentage=0.8)

    # Prepare model to explain
    if args.explainer == 'retain':
        model = RETAIN(dim_input=feature_size, dim_emb=32, dropout_emb=0.5, dim_alpha=32, dim_beta=32,
                       dropout_context=0.5, dim_output=2)
        explainer = RETAINexplainer(model, args.data)
        if args.train:
            t0 = time.time()
            explainer.fit_model(train_loader, valid_loader, test_loader, plot=True, epochs=250)
            print('Total time required to train retain: ', time.time()-t0)
        else:
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'retain'))))

    else:
        model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=100)
        if args.train:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
            train_model_rt(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=140,
                           device=device, experiment='model', data=args.data)
        model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))

        if args.explainer == 'fit':
            generator = JointFeatureGenerator(feature_size, hidden_size=feature_size*3, data=args.data)
            if args.train:
                explainer = FITExplainer(model)
                explainer.fit_generator(generator, train_loader, valid_loader)
            else:
                generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'joint_generator'))))
                explainer = FITExplainer(model, generator)

        elif args.explainer == 'integrated_gradient':
            explainer = IGExplainer(model)

        elif args.explainer == 'deep_lift':
            explainer = DeepLiftExplainer(model)

        elif args.explainer == 'feature_occlusion':
            explainer = FOExplainer(model)

        elif args.explainer == 'gradient_shap':
            explainer = GradientShapExplainer(model)
        elif args.explainer == 'FFC':

            generator = JointFeatureGenerator(feature_size, hidden_size=feature_size*3, data=args.data)
            if args.train:
                explainer = FFCExplainer(model)
                explainer.fit_generator(generator, train_loader, valid_loader)
            else:
                generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'joint_generator'))))
                explainer = FFCExplainer(model, generator)
        else:
            raise ValueError('%s explainer not defined!'%args.explainer)


    importance_scores = []
    n_samples=50
    for x,y in test_loader:
        model.train()
        model.to(device)
        x = x.to(device)
        y = y.to(device)
        with open(os.path.join(data_path, 'state_dataset_importance_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)

        score = explainer.attribute(x[:n_samples], y[:n_samples, -1].long())
        #score = explainer.attribute(x, y[:, -1].long())
        importance_scores.append(score)

        importance_scores = np.concatenate(importance_scores, 0)
        with open(os.path.join(output_path, '%s_test_importance_scores.pkl'%args.explainer), 'wb') as f:
            pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)


        # Print results
        plot_id = 1
        f, axs = plt.subplots(2)
        t = np.arange(1, gt_importance_test[plot_id].shape[-1])
        shade_state(gt_importance_test[plot_id], t, axs[0])
        for i, ref_ind in enumerate(range(x[plot_id].shape[0])):
            axs[0].plot(x[plot_id, ref_ind, 1:].cpu().numpy(), linewidth=3, label='feature %d'%(i))
            axs[1].plot(score[plot_id, ref_ind, 1:], linewidth=3, label='importance %d'%(i))
        axs[0].tick_params(axis='both', labelsize=36)
        axs[1].tick_params(axis='both', labelsize=36)
        axs[0].grid()
        f.set_figheight(20)
        f.set_figwidth(60)
        plt.subplots_adjust(hspace=.5)
        plt.savefig(os.path.join(plot_path, '%s_example.pdf' % (args.explainer)), dpi=300, orientation='landscape')
        fig_legend = plt.figure(figsize=(13, 1.2))
        handles, labels = axs[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
        fig_legend.savefig(os.path.join(plot_path, '%s_example_legend.pdf' % (args.explainer)), dpi=300, bbox_inches='tight')

        explainer_score = importance_scores.flatten()
        gt_score = gt_importance_test[:n_samples].flatten()

        print('auc:' ,metrics.roc_auc_score(gt_score,explainer_score), ' aupr:', metrics.average_precision_score(gt_score,explainer_score))
        break
