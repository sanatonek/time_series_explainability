import os
import pickle as pkl
import numpy as np
import argparse
import matplotlib.pyplot as plt

from sklearn import metrics
import torch

from TSX.models import StateClassifier, RETAIN
from TSX.utils import load_data

top_patients = [1534, 3734, 82, 3663, 3509, 870, 3305, 1484, 2604, 1672, 2733, 1057, 2599, 3319, 1239, 1671, 3095, 3783, 1935, 720, 1961, 3476, 262, 816, 2268, 723, 4469, 3818, 4126, 1575, 1526, 1457, 4542, 2015, 2512, 1419, 1749, 3822, 466, 165, 306, 1922, 1973, 1218, 1987, 701, 3344, 2285, 2363, 1429, 808, 3266, 3643, 8, 4528, 156, 229, 2684, 3588, 532, 436, 2934, 503, 2635, 4077, 2112, 2776, 2012, 2724, 420, 2978, 4265, 832, 309, 3748, 1260, 1294, 1423, 2787, 1012, 2177, 1335, 53, 2054, 2135, 4266, 3379, 379, 1580, 1720, 4409, 415, 4273, 3927, 3226, 2316, 1933, 3442, 3047, 1219, 1308, 614, 3115, 1237, 2191, 838, 3367, 1751, 2362, 3180, 2800, 2871, 3168, 3839, 4153, 7, 1014, 4428, 1803, 766, 494, 3184, 3179, 2004, 3450, 3586, 2460, 429, 1547, 1630, 1586, 4090, 2781, 2108, 1849, 4278, 2820, 1799, 1936, 1895, 1741, 4015, 3373, 973, 2291, 3122, 2979, 3377, 3892, 3742, 508, 155, 1122, 1919, 708, 1909, 3950, 4236, 3797, 4403, 4220, 3779, 1954, 3754, 3174, 2850, 2303, 1375, 1431, 67, 1278, 1757, 789, 3716, 2666, 2145, 658, 2270, 3829, 3600, 811, 3334, 79, 3803, 4131, 300, 3026, 2013, 3064, 4369, 1174, 1857, 55, 3156, 2732, 1573, 4423, 3856, 2882, 831, 2933, 3325, 1994, 440, 3788, 3126, 434, 1777, 3717, 3067, 4253, 4301, 3380, 4584, 1307, 1043, 1786, 670, 3644, 1524, 4489, 1886, 3258, 1115, 2394, 3008, 364, 4065, 3515, 2348, 2141, 3060, 1642, 1868, 4575, 4271, 967, 834, 1906, 2836, 2138, 3641, 30, 1611, 4360, 3472, 3117, 3732, 1728, 4537, 3154, 4513, 4474, 453, 3002, 4103, 4348, 2745, 3510, 299, 2157, 2718, 4127, 1811, 2523, 261, 4337, 2541, 2244, 3158, 1236, 589, 285, 2445, 4413, 893, 2272, 2422, 3639, 4556, 1067, 907, 350, 2491, 3841, 3876, 3921, 2117, 39, 2788, 179, 3314, 1083, 2038, 1776, 4356, 2926, 3786, 3323, 147]


def main(args):
    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
        with open(os.path.join(data_path, 'state_dataset_x_test.pkl'), 'rb') as f:
            x_test = pkl.load(f)
        with open(os.path.join(data_path, 'state_dataset_y_test.pkl'), 'rb') as f:
            y_test = pkl.load(f)
    elif args.data == 'simulation_l2x':
        feature_size = 3
        data_path = './data/simulated_data_l2x'
        with open(os.path.join(data_path, 'state_dataset_x_test.pkl'), 'rb') as f:
            x_test = pkl.load(f)
        with open(os.path.join(data_path, 'state_dataset_y_test.pkl'), 'rb') as f:
            y_test = pkl.load(f)
    elif args.data == 'mimic':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path='./data')
        feature_size = p_data.feature_size

        ## Select patients with varying state
        # span = []
        # testset = list(test_loader.dataset)
        # model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=200)
        # model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))
        # for i,(signal,label) in enumerate(testset):
        #    model.to(device)
        #    model.eval()
        #    risk=[]
        #    for t in range(1,48):
        #         pred = torch.nn.Softmax(-1)(model(torch.Tensor(signal[:, 0:t]).unsqueeze(0).to(device)))[:, 1]
        #         risk.append(pred.item())
        #    span.append((i,max(risk) - min(risk)))
        # span.sort(key= lambda pair:pair[1], reverse=True)
        # print([xx[0] for xx in span[0:300]])
        # print([xx[1] for xx in span[0:300]])
        # top_patients = [xx[0] for xx in span[0:300]]

        testset = list(test_loader.dataset)
        if args.percentile:
            top_patients = list(range(len(testset)))
        x_test = torch.stack(([x[0] for x_ind, x in enumerate(testset) if x_ind in top_patients])).cpu().numpy()
        y_test = torch.stack(([x[1] for x_ind, x in enumerate(testset) if x_ind in top_patients])).cpu().numpy()


    # importance_path = '/scratch/gobi2/projects/tsx/new_results/%s' % args.data
    importance_path = os.path.join(args.path, args.data)

    auc_drop, aupr_drop = [], []
    for cv in [0, 1, 2]:
        with open(os.path.join(importance_path, '%s_test_importance_scores_%s.pkl' % (args.explainer, str(cv))),
                  'rb') as f:
            importance_scores = pkl.load(f)
        if args.data=='mimic':
            importance_scores = importance_scores[top_patients]

        #### Plotting
        plot_id = 10
        pred_batch_vec = []
        plot_path = './plots/mimic/'
        t_len = importance_scores[plot_id].shape[-1]
        t = np.arange(1, t_len)
        model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=200)
        model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))
        model.eval()
        for tt in t:
            pred_tt = model(torch.Tensor(x_test[plot_id, :, :tt + 1]).unsqueeze(0)).detach().cpu().numpy()
            pred_tt = pred_tt[:,-1]
            pred_batch_vec.append(pred_tt)
        f, axs = plt.subplots(2)

        for i, ref_ind in enumerate(range(x_test[plot_id].shape[0])):
            axs[0].plot(t, x_test[plot_id, ref_ind, 1:], linewidth=3, label='feature %d' % (i))
            axs[1].plot(t, importance_scores[plot_id, ref_ind, 1:], linewidth=3, label='importance %d' % (i))

        axs[0].plot(t, pred_batch_vec, '--', linewidth=3, c='black')
        # axs[0].plot(t, y[plot_id, 1:].cpu().numpy(), '--', linewidth=3, c='red')
        axs[0].tick_params(axis='both', labelsize=36)
        axs[1].tick_params(axis='both', labelsize=36)
        axs[0].margins(0.03)
        axs[1].margins(0.03)

        # axs[0].grid()
        f.set_figheight(80)
        f.set_figwidth(120)
        plt.subplots_adjust(hspace=.5)
        name = args.explainer + '_' + args.generator_type if args.explainer == 'fit' else args.explainer
        plt.savefig(os.path.join(plot_path, '%s_example.pdf' % name), dpi=300, orientation='landscape')
        fig_legend = plt.figure(figsize=(13, 1.2))
        handles, labels = axs[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
        fig_legend.savefig(os.path.join(plot_path, '%s_example_legend.pdf' % name), dpi=300, bbox_inches='tight')

        if args.explainer=='retain':
            model = RETAIN(dim_input=feature_size, dim_emb=128, dropout_emb=0.4, dim_alpha=8, dim_beta=8,
                           dropout_context=0.4, dim_output=2)
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'retain'))))
        else:
            model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=200)
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))
            model.eval()

        min_t = 10#25
        max_t = 40
        n_drops = args.n_drops

        y1, y2, label = [], [], []
        q = np.percentile(importance_scores[:, :, min_t:], 95)
        for i,x in enumerate(x_test):
            if (args.explainer=='deep_lift' or args.explainer=='integrated_gradient' or args.explainer=='gradient_shap'):
                importance_scores = np.abs(importance_scores)
            if "mimic" in args.data:
                x_cf = x.copy()
                if args.time_imp:
                    # imp = np.unravel_index(importance_scores[i, :, min_t:].argmax(), importance_scores[i, :, min_t:].shape)
                    # x_cf = x[:,:imp[1] + min_t].copy()
                    for _ in range(n_drops):
                        imp = np.unravel_index(importance_scores[i, :, min_t:max_t].argmax(), importance_scores[i, :, min_t:max_t].shape)
                        importance_scores[i, :, imp[1] + min_t:] = -1
                        x_cf = x_cf[:,:imp[1] + min_t]
                else:
                    if args.percentile:
                        min_t_feat = [np.min(np.where(importance_scores[i, f, min_t:] >= q)[0]) if
                                      len(np.where(importance_scores[i, f, min_t:] >= q)[0]) > 0 else
                                      x.shape[-1] - min_t - 1 for f in range(p_data.feature_size)]
                        for f in range(importance_scores[i].shape[0]):
                            x_cf[f, min_t_feat[f] + min_t:] = x_cf[f, min_t_feat[f] + min_t - 1]
                    else:
                        for _ in range(n_drops):
                            imp = np.unravel_index(importance_scores[i, :, min_t:].argmax(), importance_scores[i, :, min_t:].shape)
                            importance_scores[i, imp[0], imp[1] + min_t:] = -1
                            x_cf[imp[0], imp[1] + min_t:] = x_cf[imp[0], imp[1] + min_t-1]
                label.append(y_test[i])
                if args.explainer=='retain':
                    x_t = torch.Tensor(x).unsqueeze(0).permute(0, 2, 1)
                    x_cf_t = torch.Tensor(x_cf).unsqueeze(0).permute(0, 2, 1)
                    y, _, _ = (model(x_t, torch.ones((1,)) * x_t.shape[1]))
                    y1.append(torch.nn.Softmax(-1)(y)[0,1].detach().cpu().numpy())
                    y, _, _ = (model(x_cf_t, torch.ones((1,)) * x_cf_t.shape[1]))
                    y2.append(torch.nn.Softmax(-1)(y)[0,1].detach().cpu().numpy())
                else:
                    y = torch.nn.Softmax(-1)(model(torch.Tensor(x).unsqueeze(0)))[:, 1]  # Be careful! This is fixed for class 1
                    y1.append(y.detach().cpu().numpy())
                    y = torch.nn.Softmax(-1)(model(torch.Tensor(x_cf).unsqueeze(0)))[:, 1]
                    y2.append(y.detach().cpu().numpy())

            else:
                imp = np.unravel_index(importance_scores[i,:,min_t:].argmax(), importance_scores[i,:,min_t:].shape)
                if importance_scores[i,imp[0], imp[1]+ min_t]<0:
                    continue
                else:
                    sample = x[:, :imp[1] + min_t + 1]
                    x_cf = sample.copy()
                    x_cf[imp[0], -1] = x_cf[imp[0], -2]
                    label.append(y_test[i,imp[1]+min_t])
                    lengths = (torch.ones((1,)) * x_cf.shape[1])
                    if args.explainer == 'retain':
                        x_t = torch.Tensor(sample).unsqueeze(0).permute(0, 2, 1)
                        x_cf_t = torch.Tensor(x_cf).unsqueeze(0).permute(0, 2, 1)
                        y, _, _ = (model(x_t, lengths))
                        y1.append(torch.nn.Softmax(-1)(y)[0, 1].detach().cpu().numpy())
                        y, _, _ = (model(x_cf_t, lengths))
                        y2.append(torch.nn.Softmax(-1)(y)[0, 1].detach().cpu().numpy())
                    else:
                        y = torch.nn.Softmax(-1)(model(torch.Tensor(sample).unsqueeze(0)))[:, 1] # Be careful! This is fixed for class 1
                        y1.append(y.detach().cpu().numpy())

                        y = torch.nn.Softmax(-1)(model(torch.Tensor(x_cf).unsqueeze(0)))[:, 1]
                        y2.append(y.detach().cpu().numpy())

        original_auc = metrics.roc_auc_score(np.array(label), np.array(y1))
        modified_auc = metrics.roc_auc_score(np.array(label), np.array(y2))

        original_acc = metrics.accuracy_score(np.array(label), np.round(np.array(y1)))
        modified_acc = metrics.accuracy_score(np.array(label), np.round(np.array(y2)))

        auc_drop.append(original_auc-modified_auc)
        aupr_drop.append(original_acc-modified_acc)
    print('auc: %.3f +- %.3f' % (np.mean(auc_drop), np.std(auc_drop)),
          ' auprc: %.3f +- %.3f' % (np.mean(aupr_drop), np.std(aupr_drop)))


if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run AUC drop test')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--generator_type', type=str, default='history')
    parser.add_argument('--n_drops', type=int, default=1)
    parser.add_argument('--percentile', action='store_true')
    parser.add_argument('--time_imp', action='store_true')
    parser.add_argument('--path', type=str, default='/scratch/gobi1/sana/TSX_results/new_results/')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)