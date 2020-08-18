import os
import pickle as pkl
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from TSX.models import StateClassifier, ConvClassifier, EncoderRNN, StateClassifierMIMIC, RETAIN
from TSX.utils import load_data

TOP_PATIENTS = [1534, 3734, 82, 3663, 3509, 870, 3305, 1484, 2604, 1672, 2733, 1057, 2599, 3319, 1239, 1671, 
3095, 3783, 1935, 720, 1961, 3476, 262, 816, 2268, 723, 4469, 3818, 4126, 1575, 1526, 1457, 4542, 2015, 2512, 
1419, 1749, 3822, 466, 165, 306, 1922, 1973, 1218, 1987, 701, 3344, 2285, 2363, 1429, 808, 3266, 3643, 8, 4528, 
156, 229, 2684, 3588, 532, 436, 2934, 503, 2635, 4077, 2112, 2776, 2012, 2724, 420, 2978, 4265, 832, 309, 3748, 1260, 
1294, 1423, 2787, 1012, 2177, 1335, 53, 2054, 2135, 4266, 3379, 379, 1580, 1720, 4409, 415, 4273, 3927, 3226, 2316, 
1933, 3442, 3047, 1219, 1308, 614, 3115, 1237, 2191, 838, 3367, 1751, 2362, 3180, 2800, 2871, 3168, 3839, 4153, 7, 1014, 
4428, 1803, 766, 494, 3184, 3179, 2004, 3450, 3586, 2460, 429, 1547, 1630, 1586, 4090, 2781, 2108, 1849, 4278, 2820, 1799, 1936, 
1895, 1741, 4015, 3373, 973, 2291, 3122, 2979, 3377, 3892, 3742, 508, 155, 1122, 1919, 708, 1909, 3950, 4236, 3797, 4403, 4220, 3779, 
1954, 3754, 3174, 2850, 2303, 1375, 1431, 67, 1278, 1757, 789, 3716, 2666, 2145, 658, 2270, 3829, 3600, 811, 3334, 79, 3803, 4131, 300, 
3026, 2013, 3064, 4369, 1174, 1857, 55, 3156, 2732, 1573, 4423, 3856, 2882, 831, 2933, 3325, 1994, 440, 3788, 3126, 434, 1777, 3717, 3067, 
4253, 4301, 3380, 4584, 1307, 1043, 1786, 670, 3644, 1524, 4489, 1886, 3258, 1115, 2394, 3008, 364, 4065, 3515, 2348, 2141, 3060, 
1642, 1868, 4575, 4271, 967, 834, 1906, 2836, 2138, 3641, 30, 1611, 4360, 3472, 3117, 3732, 1728, 4537, 3154, 4513, 4474, 453, 3002, 
4103, 4348, 2745, 3510, 299, 2157, 2718, 4127, 1811, 2523, 261, 4337, 2541, 2244, 3158, 1236, 589, 285, 2445, 4413, 893, 2272, 2422, 
3639, 4556, 1067, 907, 350, 2491, 3841, 3876, 3921, 2117, 39, 2788, 179, 3314, 1083, 2038, 1776, 4356, 2926, 3786, 3323, 147]


feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
                     'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP' , 'DiasBP' , 'MeanBP' ,
                     'RespRate' , 'SpO2' , 'Glucose','Temp']


select_examples = [5, 8, 10, 40]
def main(args):
    if args.data == 'simulation':
        feature_size = 3
        data_path = './data/simulated_data'
        n_classes=2
    elif args.data == 'simulation_spike':
        feature_size = 3
        data_path = './data/simulated_spike_data'
        n_classes = 2 # use with state-classifier
        if args.explainer=='retain':
            args.multiclass=True
    elif args.data == 'mimic_int':
        timeseries_feature_size = len(feature_map_mimic)
        n_classes = 4
        task = 'intervention'
        data_path = '/scratch/gobi2/projects/tsx'
        args.multiclass=True
    elif args.data == 'simulation_l2x':
        feature_size = 3
        data_path = './data/simulated_data_l2x'
        n_classes = 2
    elif args.data == 'mimic':
        timeseries_feature_size = len(feature_map_mimic)
        n_classes = 2
        task = 'mortality'
        data_path = '/scratch/gobi2/projects/tsx'
        args.multiclass=True

    if args.data == 'mimic' or args.data=='mimic_int':
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path=data_path,task=task,cv=0,train_pc=1.)
        feature_size = p_data.feature_size
        x_test = torch.stack(([x[0] for x in list(test_loader.dataset)])).cpu().numpy()
        y_test = torch.stack(([x[1] for x in list(test_loader.dataset)])).cpu().numpy()
        if args.explainer=='lime':
            x_test = x_test[:300]
            y_test = y_test[:300]
    else:
        if args.data=='simulation_l2x' or args.data=='simulation':
            file_name = 'state_dataset_'
        else:
            file_name = ''
        with open(os.path.join(data_path, file_name + 'x_test.pkl'), 'rb') as f:
            x_test = pkl.load(f)
        with open(os.path.join(data_path, file_name +'y_test.pkl'), 'rb') as f:
            y_test = pkl.load(f)

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
    else:
        top_patients = TOP_PATIENTS

    x_test = torch.stack(([x[0] for x_ind, x in enumerate(testset) if x_ind in top_patients])).cpu().numpy()
    y_test = torch.stack(([x[1] for x_ind, x in enumerate(testset) if x_ind in top_patients])).cpu().numpy()


    # importance_path = '/scratch/gobi2/projects/tsx/new_results/%s' % args.data
    importance_path = os.path.join(args.path, args.data)
    #importance_path = '/scratch/gobi2/projects/tsx/new_results/%s' % args.data
    #importance_path = '/scratch/gobi1/shalmali/TSX_results/new_results/%s' % args.data

    if args.data=='simulation_spike':
        activation = torch.nn.Sigmoid()
        if args.explainer=='retain':
            activation = torch.nn.Softmax(-1)
    elif args.data=='mimic_int':
        activation = torch.nn.Sigmoid()
        if args.explainer=='retain':
            raise ValueError('%s explainer not defined for mimic-int!' % args.explainer)
    else:
        activation = torch.nn.Softmax(-1)
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

    if args.data=='simulation_spike':
        activation = torch.nn.Sigmoid()
        if args.explainer=='retain':
            activation = torch.nn.Softmax(-1)
    elif args.data=='mimic_int':
        activation = torch.nn.Sigmoid()
        if args.explainer=='retain':
            raise ValueError('%s explainer not defined for mimic-int!' % args.explainer)
    else:
        activation = torch.nn.Softmax(-1)

    auc_drop, aupr_drop = [], []
    for cv in [0, 1, 2]:
    #for cv in [0]:
        with open(os.path.join(importance_path, '%s_test_importance_scores_%s.pkl' % (args.explainer, str(cv))),
                  'rb') as f:
            importance_scores = pkl.load(f)

        if args.data=='simulation_spike':
            model = EncoderRNN(feature_size=feature_size, hidden_size=50, regres=True, return_all=False, data=args.data, rnn="GRU")
        elif args.data=='mimic_int':
            model = StateClassifierMIMIC(feature_size=feature_size, n_state=n_classes, hidden_size=128,rnn='LSTM')
        elif args.data=='mimic' or args.data=='simulation' or args.data=='simulation_l2x':
            model = StateClassifier(feature_size=feature_size, n_state=n_classes, hidden_size=200,rnn='GRU')
        model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (args.data, 'model',cv))))

        #### Plotting
        plot_id = 10
        pred_batch_vec = []
        plot_path = './plots/mimic/'
        t_len = importance_scores[plot_id].shape[-1]
        t = np.arange(1, t_len)
        #model = StateClassifier(feature_size=feature_size, n_state=2, hidden_size=200)
        #model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (args.data, 'model'))))
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

        if args.explainer == 'retain':
            if args.data=='mimic_int':
                model = RETAIN(dim_input=feature_size, dim_emb=32, dropout_emb=0.4, dim_alpha=16, dim_beta=16,
                       dropout_context=0.4, dim_output=n_classes)
            elif args.data=='simulation_spike':
                model = RETAIN(dim_input=feature_size, dim_emb=4, dropout_emb=0.4, dim_alpha=16, dim_beta=16,
                       dropout_context=0.4, dim_output=n_classes)
            else:
                model = RETAIN(dim_input=feature_size, dim_emb=128, dropout_emb=0.4, dim_alpha=8, dim_beta=8,
                           dropout_context=0.4, dim_output=2)
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (args.data, 'retain', args.cv))))
        
        model.to(device)
        model.eval()

        if args.subpop:
            span = []
            testset = list(test_loader.dataset)
            for i,(signal,label) in enumerate(testset):
               model.to(device)
               model.eval()
               risk=[]
               if args.data=='mimic':
                    for t in range(1,signal.shape[-1]):
                        pred = activation(model(torch.Tensor(signal[:, 0:t]).unsqueeze(0).to(device)))[:, 1]
                        risk.append(pred.item())
                    span.append((i,max(risk) - min(risk)))
               elif args.data=='mimic_int':
                    for t in range(1,signal.shape[-1]):
                        pred = activation(model(torch.Tensor(signal[:, 0:t]).unsqueeze(0).to(device)))
                        risk.append(pred.detach().cpu().numpy().flatten())
                    span.append((i,np.mean(np.max(risk,0) - np.min(risk,0))))

            span.sort(key= lambda pair:pair[1], reverse=True)
            with open('shift_subsets.pkl','wb') as f:
                pkl.dump(span,f)

            with open('shift_subsets.pkl','rb') as f:
                span = pkl.load(f)

            if args.data=='mimic_int':
                top_patients = [xx[0] for xx in span if xx[1]>0.20]
            elif args.data=='mimic':
                top_patients = [xx[0] for xx in span if xx[1]>0.87]
                top_patients = [xx[0] for xx in span[0:300]]
            #print([xx[1] for xx in span[0:300]])

            if args.explainer =='lime' and args.data=='mimic_int':
                top_patients = [tt for tt in top_patients if tt<300]

            x_test = x_test[top_patients]
            y_test = y_test[top_patients]
            importance_scores = importance_scores[top_patients]

        min_t = 10#25
        max_t = 40
        n_drops = args.n_drops

        y1, y2, label = [], [], []
        q = np.percentile(importance_scores[:, :, min_t:], 95)

        for i,x in enumerate(x_test):
            if (args.explainer=='deep_lift' or args.explainer=='integrated_gradient' or args.explainer=='gradient_shap'):
                importance_scores = np.abs(importance_scores)
            if args.data=='mimic':
                x_cf = x.copy()
                if args.time_imp:
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
                    y, _, _ = (model(x_t.to(device), torch.ones((1,)) * x_t.shape[1]))
                    y1.append(torch.nn.Softmax(-1)(y)[0,1].detach().cpu().numpy())
                    y, _, _ = (model(x_cf_t.to(device), torch.ones((1,)) * x_cf_t.shape[1]))
                    y2.append(torch.nn.Softmax(-1)(y)[0,1].detach().cpu().numpy())
                else:
                    y = torch.nn.Softmax(-1)(model(torch.Tensor(x).unsqueeze(0)))[:, 1]  # Be careful! This is fixed for class 1
                    y1.append(y.detach().cpu().numpy())
                    y = torch.nn.Softmax(-1)(model(torch.Tensor(x_cf).unsqueeze(0)))[:, 1]
                    y2.append(y.detach().cpu().numpy())

            elif args.data=='simulation_l2x' or args.data=='simulation':
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
                        y, _, _ = (model(x_t.to(device), lengths))
                        y1.append(torch.nn.Softmax(-1)(y)[0, 1].detach().cpu().numpy())
                        y, _, _ = (model(x_cf_t.to(device), lengths))
                        y2.append(torch.nn.Softmax(-1)(y)[0, 1].detach().cpu().numpy())
                    else:
                        y = torch.nn.Softmax(-1)(model(torch.Tensor(sample).unsqueeze(0)))[:, 1] # Be careful! This is fixed for class 1
                        y1.append(y.detach().cpu().numpy())

                        y = torch.nn.Softmax(-1)(model(torch.Tensor(x_cf).unsqueeze(0)))[:, 1]
                        y2.append(y.detach().cpu().numpy())
            elif args.data=='simulation_spike':
                imp = np.unravel_index(importance_scores[i,:,min_t:].argmax(), importance_scores[i,:,min_t:].shape)
                sample = x[:, :imp[1]+min_t+1]
                label.append(y_test[i,imp[1]+min_t])

                if args.explainer=='retain':
                    x_t = torch.Tensor(sample).unsqueeze(0).permute(0, 2, 1)                           
                    logit,_,_ = model(torch.Tensor(x_t).to(device), (torch.ones((1,)) * sample.shape[-1]))
                    y = activation(logit)[:,1]
                    y1.append(y.detach().cpu().numpy())
                    x_cf = sample.copy()
                    x_cf[imp[0],-1] = x_cf[imp[0],-2]
                    x_cf_t = torch.Tensor(x_cf).unsqueeze(0).permute(0, 2, 1)                           
                    logit,_,_ = model(torch.Tensor(x_cf_t).to(device), (torch.ones((1,)) * x_cf.shape[-1]))
                    y = activation(logit)[:,1]
                    y2.append(y.detach().cpu().numpy())
                else:
                    y = activation(model(torch.Tensor(sample).unsqueeze(0)))[0,0]
                    #print(y.shape)
                    y1.append(y.detach().cpu().numpy())
                    x_cf = sample.copy()
                    x_cf[imp[0],-1] = x_cf[imp[0],-2]
                    y = activation(model(torch.Tensor(x_cf).unsqueeze(0)))[0,0]
                    y2.append(y.detach().cpu().numpy())
            elif args.data=='mimic_int':
                x_cf = x.copy()
                if args.time_imp:
                    for _ in range(n_drops):
                        imp = np.unravel_index(importance_scores[i, :, min_t:max_t].argmax(), importance_scores[i, :, min_t:max_t].shape)
                        importance_scores[i, :, imp[1] + min_t:] = -1
                        x_cf = x_cf[:,:imp[1] + min_t]
                    lengths = (torch.ones((1,)) * x_cf.shape[1])
                else:
                    for _ in range(n_drops):
                        imp = np.unravel_index(importance_scores[i, :, min_t:].argmax(), importance_scores[i, :, min_t:].shape)
                        if importance_scores[i,imp[0], imp[1]+ min_t]<0:
                            continue
                        else:
                            importance_scores[i, imp[0], imp[1] + min_t:] = -1
                            x_cf[imp[0], imp[1] + min_t:] = x_cf[imp[0], imp[1] + min_t-1]
                label.append(y_test[i,:,x.shape[-1]-1])
                if args.explainer=='retain':
                    x_t = torch.Tensor(x).unsqueeze(0).permute(0, 2, 1)
                    x_cf_t = torch.Tensor(x_cf).unsqueeze(0).permute(0, 2, 1)
                    y, _, _ = (model(x_t.to(device), torch.ones((1,)) * x_t.shape[1]))
                    y1.append(activation(y).detach().cpu().numpy()[0,:])
                    y, _, _ = (model(x_cf_t.to(device), torch.ones((1,)) * x_cf_t.shape[1]))
                    y2.append(activation(y).detach().cpu().numpy()[0,:])
                else:
                    y = activation(model(torch.Tensor(x).unsqueeze(0)))[0,:]
                    y1.append(y.detach().cpu().numpy())
                    y = activation(model(torch.Tensor(x_cf).unsqueeze(0)))[0,:]
                    y2.append(y.detach().cpu().numpy())

        y1 = np.array(y1)#[:,0,:]
        y2 = np.array(y2)#[:,0,:]
        label = np.array(label)
        #print(y1.shape, y2.shape, label.shape)


        original_auc = metrics.roc_auc_score(label, y1,average='macro')
        modified_auc = metrics.roc_auc_score(label, y2,average='macro')

        original_aupr = metrics.average_precision_score(label, np.array(y1))
        modified_aupr = metrics.average_precision_score(label, np.array(y2))

        auc_drop.append(original_auc-modified_auc)
        aupr_drop.append(original_aupr-modified_aupr)

    print('obs drop' if not args.time_imp else 'time_drop')
    print(args.explainer, ' auc: %.3f$\\pm$%.3f'%(np.mean(auc_drop), np.std(auc_drop)), ' aupr: %.3f$\\pm$%.3f'%(np.mean(aupr_drop), np.std(aupr_drop)))


if __name__ == '__main__':
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description='Run AUC drop test')
    parser.add_argument('--explainer', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--generator_type', type=str, default='history')
    parser.add_argument('--multiclass', action='store_true', default=False)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--n_drops', type=int, default=1)
    parser.add_argument('--percentile', action='store_true')
    parser.add_argument('--path', type=str, default='/scratch/gobi1/sana/TSX_results/new_results/')
    parser.add_argument('--subpop', action='store_true', default=False)
    parser.add_argument('--time_imp', action='store_true', default=False)
    parser.add_argument('--train_pc', type=float, default=1.)
    parser.add_argument('--percentile', action='store_true')
    parser.add_argument('--path', type=str, default='/scratch/gobi1/shalmali/TSX_results/new_results/')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)
