import os
import argparse

import torch
import numpy as np
import seaborn as sns; sns.set()
import pickle as pkl
import time

from TSX.utils import load_simulated_data, train_model_rt, compute_median_rank,\
    train_model_rt_binary, train_model, load_data
from TSX.models import StateClassifier, RETAIN, EncoderRNN
from TSX.generator import JointFeatureGenerator, JointDistributionGenerator
from TSX.explainers import RETAINexplainer, FITExplainer, IGExplainer, FFCExplainer, \
    DeepLiftExplainer, GradientShapExplainer, AFOExplainer, FOExplainer, SHAPExplainer, \
    LIMExplainer, CarryForwardExplainer, MeanImpExplainer
from sklearn import metrics

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
                     'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP' , 'DiasBP' , 'MeanBP' ,
                     'RespRate' , 'SpO2' , 'Glucose','Temp']

ks = {'simulation_spike': 1, 'simulation': 3, 'simulation_l2x': 4}


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
    parser.add_argument('--cv', type=int, default=0)
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
    # output_path = '/scratch/ssd001/home/sana/time_series_explainability/results/%s' % args.data
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_path = os.path.join('./plots/%s' % args.data)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # Load data
    if args.data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path='./data', cv=args.cv)
        feature_size = p_data.feature_size
    else:
        _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, datapath=data_path,
                                                                         percentage=0.8, data_type=data_type)

    # Prepare model to explain
    if args.explainer == 'retain':
        model = RETAIN(dim_input=feature_size, dim_emb=128, dropout_emb=0.4, dim_alpha=8, dim_beta=8,
                       dropout_context=0.4, dim_output=2)
        explainer = RETAINexplainer(model, args.data)
        if args.train:
            t0 = time.time()
            explainer.fit_model(train_loader, valid_loader, test_loader, lr=1e-3, plot=True, epochs=50)
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
                if args.train_gen:
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

        elif args.explainer == 'carry_forward':
            explainer = CarryForwardExplainer(model, train_loader)

        elif args.explainer == 'mean_imp':
            explainer = MeanImpExplainer(model, train_loader)

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

        else:
            raise ValueError('%s explainer not defined!' % args.explainer)

    # Load ground truth for simulations
    if data_type == 'state':
        with open(os.path.join(data_path, 'state_dataset_importance_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)
        with open(os.path.join(data_path, 'state_dataset_states_test.pkl'), 'rb') as f:
            state_test = pkl.load(f)
        with open(os.path.join(data_path, 'state_dataset_logits_test.pkl'), 'rb') as f:
            logits_test = pkl.load(f)
    elif data_type == 'spike':
        with open(os.path.join(data_path, 'gt_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)

    importance_scores = []
    ranked_feats=[]
    n_samples = 1
    for x, y in test_loader:
        model.train()
        model.to(device)
        x = x.to(device)
        y = y.to(device)

        t0 = time.time()
        score = explainer.attribute(x, y if args.data=='mimic' else y[:, -1].long())
        ranked_features = np.array([((-(score[n])).argsort(0).argsort(0) + 1) \
                                    for n in range(x.shape[0])])  # [:ks[args.data]]
        importance_scores.append(score)
        ranked_feats.append(ranked_features)

    importance_scores = np.concatenate(importance_scores, 0)
    print('Saving file to ', os.path.join(output_path, '%s_test_importance_scores_%d.pkl' % (args.explainer, args.cv)))
    with open(os.path.join(output_path, '%s_test_importance_scores_%d.pkl' % (args.explainer, args.cv)), 'wb') as f:
        pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

    ranked_feats = np.concatenate(ranked_feats,0)
    with open(os.path.join(output_path, '%s_test_ranked_scores.pkl' % args.explainer), 'wb') as f:
        pkl.dump(ranked_feats, f, protocol=pkl.HIGHEST_PROTOCOL)

    if 'simulation' in args.data:
        gt_soft_score = np.zeros(gt_importance_test.shape)
        gt_importance_test.astype(int)
        gt_score = gt_importance_test.flatten()
        explainer_score = importance_scores.flatten()
        if args.explainer=='deep_lift' or args.explainer=='integrated_gradient' or args.explainer=='gradient_shap':
            explainer_score = np.abs(explainer_score)
        auc_score = metrics.roc_auc_score(gt_score, explainer_score)
        aupr_score = metrics.average_precision_score(gt_score, explainer_score)

        _, median_rank, _= compute_median_rank(ranked_feats, gt_soft_score, soft=True,K=4)
        print('auc:', auc_score, ' aupr:', aupr_score, 'median rank:', median_rank)
