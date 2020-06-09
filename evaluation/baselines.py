import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set()
import pickle as pkl
import time
import pandas as pd
from scipy import interpolate
import matplotlib._color_data as mcd
from matplotlib import rc, rcParams
rc('font', weight='bold')

from TSX.utils import load_simulated_data, train_model_rt, shade_state, shade_state_state_data, \
compute_median_rank, plot_heatmap_text, train_model_rt_binary, train_model_multiclass, train_model, load_data
from TSX.models import StateClassifier, RETAIN, EncoderRNN, ConvClassifier, StateClassifierMIMIC
from TSX.generator import JointFeatureGenerator, JointDistributionGenerator
from TSX.explainers import RETAINexplainer, FITExplainer, IGExplainer, FFCExplainer, \
    DeepLiftExplainer, GradientShapExplainer, AFOExplainer, FOExplainer, SHAPExplainer, LIMExplainer
from sklearn import metrics

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
intervention_list_plot = ['niv-vent', 'vent', 'vaso','other']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
                     'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP' , 'DiasBP' , 'MeanBP' ,
                     'RespRate' , 'SpO2' , 'Glucose','Temp']

color_map = ['#7b85d4','#f37738', '#83c995', '#d7369e','#859795', '#ad5b50', '#7e1e9c', '#0343df', '#033500', '#E0FF66', '#4C005C', '#191919', '#FF0010', '#2BCE48', '#FFCC99', '#808080',
             '#740AFF', '#8F7C00', '#9DCC00', '#F0A3FF', '#94FFB5', '#FFA405', '#FFA8BB', '#426600', '#005C31', '#5EF1F2',
             '#993F00', '#990000', '#990000', '#FFFF80', '#FF5005', '#FFFF00','#FF0010', '#FFCC99','#003380']

ks = {'simulation_spike': 1, 'simulation': 3, 'simulation_l2x': 4}

# from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency

if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_gen', action='store_true')
    parser.add_argument('--generator_type', type=str, default='history')
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--gt', type=str, default='true_model', help='specify ground truth score')
    parser.add_argument('--cv', type=int, default=0, help='cross validation')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 100
    activation = torch.nn.Softmax(-1)
    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
        data_type='state'
        n_classes = 2
    elif args.data == 'simulation_l2x':
        feature_size = 3
        data_path = './data/simulated_data_l2x'
        data_type='state'
        n_classes = 2
    elif args.data == 'simulation_spike':
        feature_size = 3
        data_path = './data/simulated_spike_data'
        data_type='spike'
        n_classes = 2 # use with state-classifier
        if args.explainer=='retain':
            activation = torch.nn.Softmax()
        else:
            activation = torch.nn.Sigmoid()
        batch_size = 200
    elif args.data == 'mimic':
        data_type = 'mimic'
        timeseries_feature_size = len(feature_map_mimic)
        n_classes = 2
        task='mortality'
    elif args.data == 'mimic_int':
        timeseries_feature_size = len(feature_map_mimic)
        data_type = 'real'
        n_classes = 4
        batch_size = 256
        task = 'intervention'
        #change this to softmax for suresh et al
        activation = torch.nn.Sigmoid()
        #activation = torch.nn.Softmax(-1)

    output_path = '/scratch/gobi1/shalmali/TSX_results/new_results/%s' % args.data

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_path = os.path.join('./plots/%s' % args.data)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # Load data
    if args.data == 'mimic' or args.data=='mimic_int':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=batch_size, \
            path='/scratch/gobi2/projects/tsx/',task=task,cv=args.cv)
        feature_size = p_data.feature_size
        class_weight = p_data.pos_weight
    else:
        _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=batch_size, datapath=data_path,
                                                                         percentage=0.8, data_type=data_type,cv=args.cv)

    # Prepare model to explain
    if args.explainer == 'retain':
        if args.data=='mimic' or args.data=='simulation' or args.data=='simulation_l2x':
            model = RETAIN(dim_input=feature_size, dim_emb=128, dropout_emb=0.4, dim_alpha=8, dim_beta=8,
                       dropout_context=0.4, dim_output=2)
        elif args.data=='mimic_int':
            model = RETAIN(dim_input=feature_size, dim_emb=32, dropout_emb=0.4, dim_alpha=16, dim_beta=16,
                       dropout_context=0.4, dim_output=n_classes)
        elif args.data=='simulation_spike':
            model = RETAIN(dim_input=feature_size, dim_emb=4, dropout_emb=0.4, dim_alpha=16, dim_beta=16,
                       dropout_context=0.4, dim_output=n_classes)
        explainer = RETAINexplainer(model, args.data)
        if args.train:
            t0 = time.time()
            if args.data=='mimic' or args.data=='simulation' or args.data=='simulation_l2x':
                explainer.fit_model(train_loader, valid_loader, test_loader, lr=1e-3, plot=True, epochs=50)
            else:
                explainer.fit_model(train_loader, valid_loader, test_loader, lr=1e-4, plot=True, epochs=100,cv=args.cv)
            print('Total time required to train retain: ', time.time() - t0)
        else:
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (args.data, 'retain', args.cv))))
    else:
        if not args.binary:
            if args.data=='mimic_int':
                model = StateClassifierMIMIC(feature_size=feature_size, n_state=n_classes, hidden_size=128,rnn='LSTM')
            else:
                model = StateClassifier(feature_size=feature_size, n_state=n_classes, hidden_size=200,rnn='GRU')
        else:
            model = EncoderRNN(feature_size=feature_size, hidden_size=50, regres=True, return_all=False, data=args.data, rnn="GRU")
        if args.train:
            if not args.binary:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
                if args.data=='mimic':
                    train_model(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=100,
                                device=device, experiment='model',cv=args.cv)
                elif 'simulation' in args.data:
                    train_model_rt(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, n_epochs=50,
                               device=device, experiment='model', data=args.data,cv=args.cv)
                elif args.data=='mimic_int':
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
                    if type(activation).__name__==type(torch.nn.Softmax(-1)).__name__: #suresh et al
                        train_model_multiclass(model=model, train_loader=train_loader, valid_loader=test_loader, 
                        optimizer=optimizer, n_epochs=50, device=device, experiment='model', data=args.data,num=5, 
                        loss_criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight).to(device)),cv=args.cv)
                    else:
                        train_model_multiclass(model=model, train_loader=train_loader, valid_loader=test_loader,
                        optimizer=optimizer, n_epochs=25, device=device, experiment='model', data=args.data,num=5,
                        #loss_criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight).cuda()),cv=args.cv)
                        loss_criterion=torch.nn.BCEWithLogitsLoss(),cv=args.cv)
                        #loss_criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight).cuda()),cv=args.cv)
                        #loss_criterion=torch.nn.CrossEntropyLoss(),cv=args.cv)
            else:
                #this learning rate works much better for spike data
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                if args.data=='mimic':
                    train_model(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=200,
                                device=device, experiment='model',cv=args.cv)
                else:
                    train_model_rt_binary(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=250,
                               device=device, experiment='model', data=args.data,cv=args.cv)

        model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (args.data, 'model',args.cv))))

        if args.explainer == 'fit':
            if args.generator_type=='history':
                generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)
                if args.train:
                    if args.data=='mimic_int' or args.data=='simulation_spike':
                        explainer = FITExplainer(model,activation=torch.nn.Sigmoid(),n_classes=n_classes)
                    else:
                        explainer = FITExplainer(model)
                    explainer.fit_generator(generator, train_loader, valid_loader,cv=args.cv)
                else:
                    generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (args.data, 'joint_generator',args.cv))))
                    if args.data=='mimic_int' or args.data=='simulation_spike':
                        explainer = FITExplainer(model, generator,activation=torch.nn.Sigmoid(),n_classes=n_classes)
                    else:
                        explainer = FITExplainer(model, generator)
            elif args.generator_type=='no_history':
                generator = JointDistributionGenerator(n_components=5, train_loader=train_loader)
                if args.data=='mimic_int' or args.data=='simulation_spike':
                    explainer = FITExplainer(model, generator,activation=torch.nn.Sigmoid())
                else:
                    explainer = FITExplainer(model, generator)

        elif args.explainer == 'integrated_gradient':
            if args.data=='mimic_int' or args.data=='simulation_spike':
                explainer = IGExplainer(model,activation=activation)
            else:
                explainer = IGExplainer(model)

        elif args.explainer == 'deep_lift':
            if args.data=='mimic_int' or args.data=='simulation_spike':
                explainer = DeepLiftExplainer(model, activation=activation)
            else:
                explainer = DeepLiftExplainer(model)

        elif args.explainer == 'fo':
            if args.data=='mimic_int' or args.data=='simulation_spike':
                explainer = FOExplainer(model,activation=activation)
            else:
                explainer = FOExplainer(model)

        elif args.explainer == 'afo':
            if args.data=='mimic_int' or args.data=='simulation_spike':
                explainer = AFOExplainer(model, train_loader,activation=activation)
            else:
                explainer = AFOExplainer(model, train_loader)

        elif args.explainer == 'gradient_shap':
            if args.data=='mimic_int' or args.data=='simulation_spike':
                explainer = GradientShapExplainer(model, activation=activation)
            else:
                explainer = GradientShapExplainer(model)

        elif args.explainer == 'ffc':
            generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)
            if args.train:
                if args.data=='mimic_int' or args.data=='simulation_spike':
                    explainer = FFCExplainer(model,activation=activation)
                else:
                    explainer = FFCExplainer(model)
                explainer.fit_generator(generator, train_loader, valid_loader)
            else:
                generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'joint_generator'))))
                if args.data=='mimic_int' or args.data=='simulation_spike':
                    explainer = FFCExplainer(model, generator,activation=activation)
                else:
                    explainer = FFCExplainer(model, generator)

        elif args.explainer == 'shap':
            explainer = SHAPExplainer(model, train_loader)

        elif args.explainer == 'lime':
            if args.data=='mimic_int' or args.data=='simulation_spike':
                explainer = LIMExplainer(model, train_loader, activation=activation,n_classes=n_classes)
            else:
                explainer = LIMExplainer(model, train_loader)

        elif args.explainer == 'retain':
             explainer = RETAINexplainer(model,self.data)

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

        score = explainer.attribute(x, y if (args.data=='mimic') else y[:, -1].long())
        ranked_features = np.array([((-(score[n])).argsort(0).argsort(0) + 1) \
                                    for n in range(x.shape[0])])  # [:ks[args.data]]
        importance_scores.append(score)
        ranked_feats.append(ranked_features)
 
        importance_scores = np.concatenate(importance_scores, 0)
        print('*******************\n Saving file to ',
              os.path.join(output_path, '%s_test_importance_scores_%d.pkl' % (args.explainer, args.cv)))
        print(output_path)
        with open(os.path.join(output_path, '%s_test_importance_scores_%d.pkl' % (args.explainer, args.cv)), 'wb') as f:
            pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

        if args.data=='mimic_int': #plot all label logits
            labels = np.zeros((x.shape[0], y.shape[1], x.shape[2]-1))
        else:
            labels = np.zeros((x.shape[0], x.shape[-1]-1))

        for t in range(1, x.shape[-1]):
            if args.explainer=='retain':
                x_retain = x.permute(0, 2, 1)
                p_y_t, _, _ = explainer.base_model(x_retain[:, :t + 1, :], (torch.ones((len(x),)) * t+1))
            else:
                p_y_t = explainer.base_model(x[:, :, :t + 1])
            p_y_t = activation(p_y_t)
            if args.data=='mimic' or args.data=='simulation' or args.data=='simulation_l2x':
                labels[:, t - 1] = np.array([p > 0.5 for p in p_y_t.cpu().detach().numpy()[:, 1]]).flatten()
            elif args.data=='mimic_int':
                labels[:, :, t - 1] = p_y_t.cpu().detach().numpy()
            elif args.data=='simulation_spike':
                labels[:, t - 1] = p_y_t[:,0].cpu().detach().numpy()
        break

    # Print results
    plot_id = 4

    f, axs = plt.subplots(3)
    f.set_figheight(6)
    f.set_figwidth(10)
    score_pd = pd.DataFrame(columns=['f1', 'f2', 'f3', 's1', 's2', 's3'])
    score_pd['t'] = pd.Series(np.arange(1, score[plot_id].shape[-1]))
    cmap = sns.cubehelix_palette(rot=.2, as_cmap=True)
    bottom = cm.get_cmap('Blues', 128)
    for feat in [1, 2, 3]:  # range(1,2):
        score_pd['f%d' % feat] = pd.Series(x[plot_id, feat - 1, 1:].cpu().numpy())
        score_pd['s%d' % feat] = pd.Series(score[plot_id, feat - 1, :])
        f = interpolate.interp1d(score_pd['t'], score_pd['f%d' % feat], fill_value="extrapolate")
        f_score = interpolate.interp1d(score_pd['t'], score_pd['s%d' % feat], fill_value="extrapolate")
        xnew = np.arange(1, score[plot_id].shape[-1] - 0.99, 0.01)
        ynew = f(xnew)
        score_new = f_score(xnew)
        # axs[feat-1].scatter(xnew, ynew, c=cm.hot(score_new/2.+0.5), edgecolor='none')
        axs[feat - 1].scatter(xnew, ynew, c=bottom(score_new / 2. + 0.5), edgecolor='none')
    plt.legend()
    plt.savefig(os.path.join(plot_path, 'new_viz.pdf'), dpi=300, orientation='landscape')

    t_len = score[plot_id].shape[-1]
    f, axs = plt.subplots(3)

    #plot_heatmap_text(ranked_features[plot_id, :, 1:], score[plot_id, :, 1:],
    #                  os.path.join(plot_path, '%s_example_heatmap.pdf' % args.explainer), axs[2])
    t = np.arange(1, t_len)

    if data_type == 'state':
        shade_state_state_data(state_test[plot_id], t, axs[0])

    if args.data=='mimic_int':
        imp_rank = np.flip(np.argsort(score[plot_id,5:].max(1)).flatten())
        plot_idx = imp_rank[:5]
    else:
        plot_idx = list(range(x.shape[1]))

    for i, ref_ind in enumerate(plot_idx):
        axs[1].plot(t, x[plot_id, ref_ind, 1:].cpu().numpy(), linewidth=3, label= 'feature ' + str(ref_ind), color=color_map[i])
        axs[2].plot(t, score[plot_id, ref_ind, 1:], linewidth=3, color=color_map[i],linestyle='-')

    if args.data=='mimic_int':
        for nn in range(n_classes):
            axs[0].plot(t, labels[plot_id, nn, :], '--', linewidth=3, c=color_map[-nn-1],label=intervention_list_plot[nn])
    else:
        axs[0].plot(t, labels[plot_id, :], '--', linewidth=3, c=color_map[-1],label='outcome')

    axs[0].tick_params(axis='both', labelsize=20)
    axs[2].tick_params(axis='both', labelsize=20)
    axs[1].tick_params(axis='both', labelsize=20)
    axs[1].set_title('%s'%args.explainer, fontsize=30)
    axs[0].margins(0.03)
    axs[2].margins(0.03)
    axs[1].margins(0.03)
    axs[0].grid(True,linestyle=':')
    axs[1].grid(True,linestyle=':')
    axs[2].grid(True,linestyle=':')

    f.set_figheight(16)
    f.set_figwidth(24)
    plt.subplots_adjust(hspace=.5)
    name = args.explainer + '_' + args.generator_type if args.explainer == 'fit' else args.explainer
    plt.savefig(os.path.join(plot_path, '%s_example.pdf' % name), dpi=300, orientation='landscape')
    fig_legend = plt.figure(figsize=(13, 1.2))
    handles, labels = axs[1].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
    fig_legend.savefig(os.path.join(plot_path, '%s_example_legend.pdf' % name), dpi=300, bbox_inches='tight')

    ranked_feats = np.concatenate(ranked_feats,0)
    with open(os.path.join(output_path, '%s_test_ranked_scores.pkl' % args.explainer), 'wb') as f:
        pkl.dump(ranked_feats,f,protocol=pkl.HIGHEST_PROTOCOL)

    if 'simulation' in args.data:
        gt_soft_score = np.zeros(gt_importance_test.shape)
        gt_importance_test.astype(int)
        gt_score = gt_importance_test.flatten()
        explainer_score = importance_scores.flatten()
        if (args.explainer=='deep_lift' or args.explainer=='integrated_gradient' or args.explainer=='gradient_shap'):
            explainer_score = np.abs(explainer_score)
        auc_score = metrics.roc_auc_score(gt_score, explainer_score)
        aupr_score = metrics.average_precision_score(gt_score, explainer_score)

        _, median_rank, _= compute_median_rank(ranked_feats, gt_soft_score, soft=True,K=4)
        print('auc:', auc_score, ' aupr:', aupr_score, 'median rank:', median_rank)
