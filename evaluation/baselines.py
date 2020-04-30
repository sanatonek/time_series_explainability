import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import time

from TSX.utils import load_simulated_data, train_model_rt, shade_state, shade_state_state_data, compute_median_rank, plot_heatmap_text
from TSX.models import StateClassifier, RETAIN
from TSX.generator import JointFeatureGenerator
from TSX.explainers import RETAINexplainer, FITExplainer, IGExplainer, FFCExplainer, DistGenerator, \
    DeepLiftExplainer, GradientShapExplainer, AFOExplainer, FOExplainer, SHAPExplainer, LIMExplainer
from sklearn import metrics

ks = {'simulation_spike': 1, 'simulation': 3, 'simulation_l2x': 4}

# from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency

if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gt', type=str, default='true_model', help='specify ground truth score')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
    elif args.data == 'simulation_l2x':
        feature_size = 10
        data_path = './data/simulated_data_l2x'

    output_path = '/scratch/gobi1/shalmali/TSX_results/new_results/%s' % args.data
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_path = os.path.join('./plots/%s' % args.data)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # Load data
    _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, datapath=data_path,
                                                                     percentage=0.8)

    # Prepare model to explain
    if args.explainer == 'retain':
        model = RETAIN(dim_input=feature_size, dim_emb=32, dropout_emb=0.5, dim_alpha=32, dim_beta=32,
                       dropout_context=0.5, dim_output=2)
        explainer = RETAINexplainer(model, args.data)
        if args.train:
            t0 = time.time()
            explainer.fit_model(train_loader, valid_loader, test_loader, plot=True, epochs=250)
            print('Total time required to train retain: ', time.time() - t0)
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
            generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)
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

        elif args.explainer == 'afo':
            explainer = AFOExplainer(model, train_loader)

        elif args.explainer == 'gradient_shap':
            explainer = GradientShapExplainer(model)

        elif args.explainer == 'FFC':
            generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)
            if args.train:
                explainer = FFCExplainer(model)
                explainer.fit_generator(generator, train_loader, valid_loader)
            else:
                generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'joint_generator'))))
                explainer = FFCExplainer(model, generator)

        elif args.explainer == 'shap':
            explainer = SHAPExplainer(model, train_loader)

        elif args.explainer == 'lime':
            explainer = LIMExplainer(model, train_loader)

        elif args.explainer == 'dist':
            explainer = DistGenerator(model, train_loader, n_componenets=2)

        else:
            raise ValueError('%s explainer not defined!' % args.explainer)

    importance_scores = []
    ranked_feats=[]
    n_samples = 1
    for x, y in test_loader:
        model.train()
        model.to(device)
        x = x.to(device)
        y = y.to(device)
        with open(os.path.join(data_path, 'state_dataset_importance_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)
        with open(os.path.join(data_path, 'state_dataset_states_test.pkl'),'rb') as f:
            state_test = pkl.load(f)


        n_avg = 1
        for i in range(n_avg):
            if i==0:
                score = explainer.attribute(x, y[:, -1].long())
            else:
                scorez = explainer.attribute(x, y[:, -1].long())
                score +=scorez
        score /= n_avg

        # score = explainer.attribute(x, y[:, -1].long())
        ranked_features = np.array([((-(score[n])).argsort(0).argsort(0)+1) \
        for n in range(x.shape[0])])#[:ks[args.data]]
        importance_scores.append(score)
        ranked_feats.append(ranked_features)
        #print(np.any(np.isnan(gt_importance_test)), np.any(np.isnan(ranked_features)), np.any(np.isnan(score)))

        # Print results
        plot_id = 3
        t_len = gt_importance_test[plot_id].shape[-1]
        #print('gt_importance', gt_importance_test)
        #print('median scores:', np.median(ranked_features,axis=0))
        f, axs = plt.subplots(3)
        plot_heatmap_text(ranked_features[plot_id,:,1:], score[plot_id,:,1:],
                          os.path.join(plot_path, '%s_example_heatmap.pdf' % args.explainer),axs[1])
        t = np.arange(1, t_len)
        pred = []
        model.eval()
        for tt in t:
            pred_tt = model(x[plot_id, :, :tt + 1].unsqueeze(0))[0, 1].detach().cpu().numpy()
            pred.append(pred_tt)

        if args.gt == 'pred_model':
            for tt in t:
                if tt>1:
                    p_y_t = self.base_model(x[:, :, :min((tt + 1), t_len)])
                    labels = np.array([p>0.5 for p in p_y_t.cpu().detach().numpy()[:,1]]).flatten()
                    label_change = abs((labels[:,tt-1]-labels[:,tt-2]).reshape(-1,1))
                    gt_importance_test[:,:,tt-1] = np.multiply(np.repeat(label_change,x.shape[1],axis=1), \
                       gt_importance_test[:,:,tt-1])
        elif args.gt == 'true_model':
            for tt in t:
                if tt>1:
                    label_change = abs((y[:,tt-1]-y[:,tt-2]).cpu().detach().numpy().reshape(-1,1))
                    gt_importance_test[:,:,tt-1] = np.multiply(np.repeat(label_change,x.shape[1],axis=1), \
                       gt_importance_test[:,:,tt-1])

        gt_importance_test.astype(int)


        shade_state_state_data(state_test[plot_id], t, axs[0])
        for i, ref_ind in enumerate(range(x[plot_id].shape[0])):
            axs[0].plot(t, x[plot_id, ref_ind, 1:].cpu().numpy(), linewidth=3, label='feature %d' % (i))
            axs[2].plot(t, score[plot_id, ref_ind, 1:], linewidth=3, label='importance %d' % (i))
        axs[0].plot(t, pred, '--', linewidth=3, c='black')
        axs[0].tick_params(axis='both', labelsize=36)
        axs[2].tick_params(axis='both', labelsize=36)
        axs[1].tick_params(axis='both', labelsize=36)
        axs[0].margins(0.0)
        axs[2].margins(0.0)
        axs[1].margins(0.0)

        # axs[0].grid()
        f.set_figheight(120)
        f.set_figwidth(120)
        plt.subplots_adjust(hspace=.5)
        plt.savefig(os.path.join(plot_path, '%s_example.pdf' % args.explainer), dpi=300, orientation='landscape')
        fig_legend = plt.figure(figsize=(13, 1.2))
        handles, labels = axs[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
        fig_legend.savefig(os.path.join(plot_path, '%s_example_legend.pdf' % args.explainer), dpi=300,
                           bbox_inches='tight')

        # explainer_score = score.flatten()
        # gt_score = gt_importance_test[:n_samples].flatten()
        #
        # print('auc:' ,metrics.roc_auc_score(gt_score,explainer_score), ' aupr:', metrics.average_precision_score(
        # gt_score,explainer_score))


    importance_scores = np.concatenate(importance_scores, 0)
    ranked_feats = np.concatenate(ranked_feats,0)
    with open(os.path.join(output_path, '%s_test_importance_scores.pkl' % args.explainer), 'wb') as f:
        pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path, '%s_test_ranked_scores.pkl' % args.explainer), 'wb') as f:
        pkl.dump(ranked_feats,f,protocol=pkl.HIGHEST_PROTOCOL)

    explainer_score = importance_scores.flatten()
    gt_score = gt_importance_test.flatten()
    print(gt_score.shape, explainer_score.shape)

    auc_score = metrics.roc_auc_score(gt_score, explainer_score)
    aupr_score = metrics.average_precision_score(gt_score, explainer_score)
    _, median_rank, _= compute_median_rank(ranked_feats, gt_importance_test)
    # fdr = fp /(fp + tp)
    print('auc:', auc_score, ' aupr:', aupr_score, 'median rank:', median_rank)
    #break
