import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pickle as pkl
import time
import pandas as pd
from scipy import interpolate

from TSX.utils import load_simulated_data, train_model_rt, shade_state, shade_state_state_data, compute_median_rank,\
    plot_heatmap_text, train_model_rt_binary, train_model, load_data
from TSX.models import StateClassifier, RETAIN, EncoderRNN #, TrueClassifier
from TSX.generator import JointFeatureGenerator, JointDistributionGenerator
from TSX.explainers import RETAINexplainer, FITExplainer, IGExplainer, FFCExplainer, \
    DeepLiftExplainer, GradientShapExplainer, AFOExplainer, FOExplainer, SHAPExplainer, LIMExplainer
from sklearn import metrics

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
                     'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP' , 'DiasBP' , 'MeanBP' ,
                     'RespRate' , 'SpO2' , 'Glucose','Temp']

ks = {'simulation_spike': 1, 'simulation': 3, 'simulation_l2x': 4}

# from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency

if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--generator_type', type=str, default='history')
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--gt', type=str, default='true_model', help='specify ground truth score')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
        data_type='state'
    elif args.data == 'simulation_l2x':
        feature_size = 3
        data_path = './data/simulated_data_l2x'
        data_type='state'
    elif args.data == 'simulation_spike':
        feature_size = 3
        data_path = './data/simulated_spike_data'
        data_type='spike'
    elif args.data == 'mimic':
        data_type = 'mimic'
        timeseries_feature_size = len(feature_map_mimic)

    output_path = '/scratch/gobi1/sana/TSX_results/new_results/%s' % args.data
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_path = os.path.join('./plots/%s' % args.data)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # Load data
    if args.data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path='./data')
        feature_size = p_data.feature_size
    else:
        _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, datapath=data_path,
                                                                         percentage=0.8, data_type=data_type)

    # Prepare model to explain
    if args.explainer == 'retain':
        model = RETAIN(dim_input=feature_size, dim_emb=32, dropout_emb=0.4, dim_alpha=16, dim_beta=16,
                       dropout_context=0.4, dim_output=2)
        explainer = RETAINexplainer(model, args.data)
        if args.train:
            t0 = time.time()
            explainer.fit_model(train_loader, valid_loader, test_loader, lr=1e-4, plot=True, epochs=100)
            print('Total time required to train retain: ', time.time() - t0)
        else:
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'retain'))))

    else:
        if not args.binary:
            # model = TrueClassifier(feature_size=feature_size, n_state=2, hidden_size=100)
            model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=200)
        else:
            model = EncoderRNN(feature_size=feature_size, hidden_size=50, regres=True, return_all=False, data=args.data, rnn="GRU")
        if args.train:
            if not args.binary:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
                if args.data=='mimic':
                    train_model(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=100,
                                device=device, experiment='model')
                else:
                    train_model_rt(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, n_epochs=50,
                               device=device, experiment='model', data=args.data)
            else:
                #this learning rate works much better for spike data
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
                if args.data=='mimic':
                    train_model(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=200,
                                device=device, experiment='model')
                else:
                    train_model_rt_binary(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=120,
                               device=device, experiment='model', data=args.data)

        model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))

        if args.explainer == 'fit':
            if args.generator_type=='history':
                generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)
                if args.train:
                    explainer = FITExplainer(model)
                    explainer.fit_generator(generator, train_loader, valid_loader)
                else:
                    generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'joint_generator'))))
                    explainer = FITExplainer(model, generator)
            elif args.generator_type=='no_history':
                generator = JointDistributionGenerator(n_components=5, train_loader=train_loader)
                explainer = FITExplainer(model, generator)

        elif args.explainer == 'integrated_gradient':
            explainer = IGExplainer(model)

        elif args.explainer == 'deep_lift':
            explainer = DeepLiftExplainer(model)

        elif args.explainer == 'fo':
            explainer = FOExplainer(model)

        elif args.explainer == 'afo':
            explainer = AFOExplainer(model, train_loader)

        elif args.explainer == 'gradient_shap':
            explainer = GradientShapExplainer(model)

        elif args.explainer == 'ffc':
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

        # elif args.explainer == 'dist':
        #     explainer = DistGenerator(model, train_loader, n_componenets=2)

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

        if data_type == 'state':
            with open(os.path.join(data_path, 'state_dataset_importance_test.pkl'), 'rb') as f:
                gt_importance_test = pkl.load(f)
            with open(os.path.join(data_path, 'state_dataset_states_test.pkl'),'rb') as f:
                state_test = pkl.load(f)
            with open(os.path.join(data_path, 'state_dataset_logits_test.pkl'),'rb') as f:
                logits_test = pkl.load(f)
        elif data_type == 'spike':
            with open(os.path.join(data_path, 'gt_test.pkl'), 'rb') as f:
                gt_importance_test = pkl.load(f)
        else:
            pass

        score = explainer.attribute(x, y if args.data=='mimic' else y[:, -1].long())

        with open(os.path.join(output_path, '%s_test_importance_scores.pkl' % args.explainer), 'wb') as f:
            pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

        ranked_features = np.array([((-(score[n])).argsort(0).argsort(0) + 1) \
                                    for n in range(x.shape[0])])  # [:ks[args.data]]
        if args.explainer == 'fit':
            score_mean_shift = explainer.attribute(x, y[:, -1].long(), distance_metric='mean_divergence')
        importance_scores.append(score)
        ranked_feats.append(ranked_features)

        labels = np.zeros((x.shape[0], x.shape[-1]))
        for t in range(1, x.shape[-1]):
            p_y_t = explainer.base_model(x[:, :, :t + 1], return_multi=True)
            labels[:, t - 1] = np.array([p > 0.5 for p in p_y_t.cpu().detach().numpy()[:, 1]]).flatten()
        #print(np.any(np.isnan(gt_importance_test)), np.any(np.isnan(ranked_features)), np.any(np.isnan(score)))

        # Print results
        plot_id = 5

        f, axs = plt.subplots(3)
        f.set_figheight(6)
        f.set_figwidth(10)
        score_pd = pd.DataFrame(columns=['f1', 'f2', 'f3', 's1', 's2', 's3'])
        score_pd['t'] = pd.Series(np.arange(1, score[plot_id].shape[-1]))
        cmap = sns.cubehelix_palette(rot=.2, as_cmap=True)
        bottom = cm.get_cmap('Blues', 128)
        for feat in [1, 2, 3]:#range(1,2):
            score_pd['f%d'%feat] = pd.Series(x[plot_id, feat-1, 1:].cpu().numpy())
            score_pd['s%d' % feat] = pd.Series(score[plot_id, feat - 1, :])
            f = interpolate.interp1d(score_pd['t'], score_pd['f%d'%feat], fill_value="extrapolate")
            f_score = interpolate.interp1d(score_pd['t'], score_pd['s%d'%feat], fill_value="extrapolate")
            xnew = np.arange(1, score[plot_id].shape[-1]-0.99, 0.01)
            ynew = f(xnew)
            score_new = f_score(xnew)
            # axs[feat-1].scatter(xnew, ynew, c=cm.hot(score_new/2.+0.5), edgecolor='none')
            axs[feat - 1].scatter(xnew, ynew, c=bottom(score_new / 2. + 0.5), edgecolor='none')

            # g = sns.scatterplot(x=score_pd['t'], y=score_pd['f%d'%feat], hue=score_pd['s%d'%feat],
            #                     marker='o', palette=cmap, ax=axs[feat-1])
            # g.legend_.remove()
        plt.legend()
        plt.savefig(os.path.join(plot_path, 'new_viz.pdf'), dpi=300, orientation='landscape')


        t_len = score[plot_id].shape[-1]
        f, axs = plt.subplots(4 if args.explainer=='fit' else 3)

        plot_heatmap_text(ranked_features[plot_id,:,1:], score[plot_id,:,1:],
                          os.path.join(plot_path, '%s_example_heatmap.pdf' % args.explainer),axs[1])
        t = np.arange(1, t_len)

        # pred_batch_vec = []
        # model.eval()
        # for tt in t:
        #     pred_tt = model(x[plot_id, :, :tt + 1].unsqueeze(0)).detach().cpu().numpy()
        #     # pred_tt = np.argmax(pred_tt, -1)
        #     pred_tt = pred_tt[:,-1]
        #     pred_batch_vec.append(pred_tt)


        # gt_soft_score = np.zeros(gt_importance_test.shape)
        # gt_importance_test.astype(int)

        if data_type=='state':
            shade_state_state_data(state_test[plot_id], t, axs[0])
            '''
            if args.gt == 'pred_model':
                for tt in t:
                    if tt>1:
                        label_change = abs((pred_batch_vec[tt-1][:,1]-pred_batch_vec[tt-2][:,1]).reshape(-1,1))
                        gt_importance_test[:,:,tt-1] = np.multiply(np.repeat(label_change,x.shape[1],axis=1), \
                        gt_importance_test[:,:,tt-1])
            elif args.gt == 'true_model':
                for tt in t:
                    if tt>1:
                        label_change = abs((y[:,tt-1]-y[:,tt-2]).cpu().detach().numpy().reshape(-1,1))
                        gt_importance_test[:,:,tt-1] = np.multiply(np.repeat(label_change,x.shape[1],axis=1), \
                        gt_importance_test[:,:,tt-1])

                        logits_change = abs((logits_test[:,tt-1]-logits_test[:,tt-2]).reshape(-1,1))
                        gt_soft_score[:,:,tt-1] = np.multiply(np.repeat(logits_change,x.shape[1],axis=1), \
                        gt_importance_test[:,:,tt-1])
            '''
        for i, ref_ind in enumerate(range(x[plot_id].shape[0])):
            axs[0].plot(t, x[plot_id, ref_ind, 1:].cpu().numpy(), linewidth=3, label='feature %d' % (i))
            axs[2].plot(t, score[plot_id, ref_ind, 1:], linewidth=3, label='importance %d' % (i))
            if args.explainer=='fit':
                axs[3].plot(t,
                            score_mean_shift[plot_id, ref_ind, 1:], linewidth=3, label='importance %d' % (i))

        # axs[0].plot(t, pred_batch_vec, '--', linewidth=3, c='black')
        axs[0].plot(t, y[plot_id,1:].cpu().numpy(), '--', linewidth=3, c='red')
        axs[0].tick_params(axis='both', labelsize=36)
        axs[2].tick_params(axis='both', labelsize=36)
        axs[1].tick_params(axis='both', labelsize=36)
        axs[0].margins(0.03)
        axs[2].margins(0.03)
        axs[1].margins(0.03)

        # axs[0].grid()
        f.set_figheight(80)
        f.set_figwidth(120)
        plt.subplots_adjust(hspace=.5)
        name = args.explainer+'_'+args.generator_type if args.explainer=='fit' else args.explainer
        plt.savefig(os.path.join(plot_path, '%s_example.pdf' % name), dpi=300, orientation='landscape')
        fig_legend = plt.figure(figsize=(13, 1.2))
        handles, labels = axs[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
        fig_legend.savefig(os.path.join(plot_path, '%s_example_legend.pdf' %name), dpi=300, bbox_inches='tight')


    importance_scores = np.concatenate(importance_scores, 0)
    ranked_feats = np.concatenate(ranked_feats,0)

    with open(os.path.join(output_path, '%s_test_ranked_scores.pkl' % args.explainer), 'wb') as f:
        pkl.dump(ranked_feats,f,protocol=pkl.HIGHEST_PROTOCOL)

    explainer_score = importance_scores.flatten()

    if 'simulation' in args.data:
        gt_score = gt_importance_test.flatten()

        auc_score = metrics.roc_auc_score(gt_score, explainer_score)
        aupr_score = metrics.average_precision_score(gt_score, explainer_score)

        _, median_rank, _= compute_median_rank(ranked_feats, gt_soft_score, soft=True,K=4)
        # fdr = fp /(fp + tp)
        print('auc:', auc_score, ' aupr:', aupr_score, 'median rank:', median_rank)
        #break
