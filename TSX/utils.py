import numpy as np
import os
import torch
import torch.utils.data as utils
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, roc_auc_score
from TSX.models import PatientData, NormalPatientData, GHGData
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import KFold
import seaborn as sns
sns.set()


line_styles_map=['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--','-.',':']
marker_styles_map=['o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h','o','v','^','*','+','p','8','h']

# Ignore sklearn warnings caused by ill-defined precision score (caused by single class prediction)
import warnings
warnings.filterwarnings("ignore")

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']


def evaluate(labels, predicted_label, predicted_probability):
    labels_array = np.array(labels.cpu())
    prediction_array = np.array(predicted_label.cpu())
    if len(np.unique(labels_array)) < 2:
        auc = 0
    else:
        auc = roc_auc_score(np.array(labels.cpu()), np.array(predicted_probability.view(len(labels), -1).detach().cpu()))
    recall = torch.matmul(labels, predicted_label).item()
    precision = precision_score(labels_array, prediction_array)
    correct_label = torch.eq(labels, predicted_label).sum()
    return auc, recall, precision, correct_label


def test(test_loader, model, device, criteria=torch.nn.BCELoss(), verbose=True):
    model.to(device)
    correct_label = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    total = 0
    auc_test = 0
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
        out = model(x)
        y = y.view(y.shape[0],)
        prediction = (out > 0.5).view(len(y), ).float()
        auc, recall, precision, correct = evaluate(y, prediction, out)
        correct_label += correct
        auc_test = auc_test + auc
        recall_test = + recall
        precision_test = + precision
        count = + 1
        loss = + criteria(out.view(len(y), ), y).item()
        total += len(x)
    return recall_test, precision_test, auc_test/(i+1), correct_label, loss


def train(train_loader, model, device, optimizer, loss_criterion=torch.nn.BCELoss()):
    model = model.to(device)
    model.train()
    auc_train = 0
    recall_train, precision_train, auc_train, correct_label, epoch_loss = 0, 0, 0, 0, 0
    for i, (signals, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
        labels = labels.view(labels.shape[0],)
        labels = labels.view(labels.shape[0],)
        risks = model(signals)
        predicted_label = (risks > 0.5).view(len(labels), ).float()
        auc, recall, precision, correct = evaluate(labels, predicted_label, risks)
        correct_label += correct
        auc_train = auc_train + auc
        recall_train = + recall
        precision_train = + precision

        loss = loss_criterion(risks.view(len(labels), ), labels)
        epoch_loss = + loss.item()
        loss.backward()
        optimizer.step()
    return recall_train, precision_train, auc_train/(i+1), correct_label, epoch_loss, i+1


def train_model(model, train_loader, valid_loader, optimizer, n_epochs, device, experiment,data='mimic'):
    train_loss_trend = []
    test_loss_trend = []

    for epoch in range(n_epochs + 1):
        recall_train, precision_train, auc_train, correct_label_train, epoch_loss,n_batches = train(train_loader, model,
                                                                                            device, optimizer)
        recall_test, precision_test, auc_test, correct_label_test, test_loss = test(valid_loader, model,
                                                                                      device)
        train_loss_trend.append(epoch_loss)
        test_loss_trend.append(test_loss)
        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', epoch_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_train / (len(train_loader.dataset))),
                  ' AUC: %.2f' % (auc_train))
            print('Test ===>loss: ', test_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_test / (len(valid_loader.dataset))),
                  ' AUC: %.2f' % (auc_test))


    # Save model and results
    if not os.path.exists(os.path.join("./ckpt/",data)):
        os.mkdir("./ckpt/")
        os.mkdir(os.path.join("./ckpt/", data))
    if not os.path.exists(os.path.join("./plots/",data)):
        os.mkdir("./plots/")
        os.mkdir(os.path.join("./plots/", data))
    torch.save(model.state_dict(), './ckpt/' + data + '/'+ str(experiment) + '.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join('./plots', data, 'train_loss.pdf'))


def train_model_rt(model, train_loader, valid_loader, optimizer, n_epochs, device, experiment, data='simulation',num=5):
    print('training data: ', data)
    train_loss_trend = []
    test_loss_trend = []

    model.to(device)
    loss_criterion = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        recall_train, precision_train, auc_train, correct_label_train, epoch_loss, count = 0, 0, 0, 0, 0, 0
        for i, (signals,labels) in enumerate(train_loader):
            signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
            if data=='simulation':
                time_points = [int(tt) for tt in np.linspace(1,signals.shape[2]-2, num=num)]
            else:
                time_points = [int(tt) for tt in np.logspace(0, np.log10(signals.shape[2] - 1), num=num)]
            for t in time_points:
                optimizer.zero_grad()
                predictions = model(signals[:,:,:t+1])

                predicted_label = (predictions > 0.5).float()
                labels_th = (labels[:,t]>0.5).float()
                auc, recall, precision, correct = evaluate(labels_th.contiguous().view(-1), predicted_label.contiguous().view(-1), predictions.contiguous().view(-1))
                correct_label_train += correct
                auc_train += auc
                recall_train += recall
                precision_train += precision
                count += 1

                reconstruction_loss = loss_criterion(predictions, labels[:,t].to(device))
                epoch_loss += reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        test_num=num
        test_loss, recall_test, precision_test, auc_test, correct_label_test = test_model_rt(model,valid_loader,num=test_num)

        train_loss_trend.append(epoch_loss/((i+1)*num))
        test_loss_trend.append(test_loss)

        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', epoch_loss/((i+1)*num),
                  ' Accuracy: %.2f percent' % (100 * correct_label_train / (len(train_loader.dataset)*num)),
                  ' AUC: %.2f' % (auc_train/((i+1)*num)))
            print('Test ===>loss: ', test_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_test / (len(valid_loader.dataset)*test_num)),
                  ' AUC: %.2f' % (auc_test))

    test_loss, recall_test, precision_test, auc_test, correct_label_test = test_model_rt(model, valid_loader)
    print('Test loss: ', test_loss)

    # Save model and results
    if not os.path.exists(os.path.join("./ckpt/", data)):
        os.mkdir(os.path.join("./ckpt/", data))
    torch.save(model.state_dict(), './ckpt/' + data + '/' + str(experiment) + '.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join('./plots', data, 'train_loss.pdf'))


def train_model_rt_rg(model, train_loader, valid_loader, optimizer, n_epochs, device, experiment, data='ghg'):
    print('training data: ', data)
    train_loss_trend = []
    test_loss_trend = []

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_criterion = torch.nn.MSELoss()
    print('loss function: MSE')
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        num = 50
        for i, (signals,labels) in enumerate(train_loader):
            signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
            for t in [int(tt) for tt in np.linspace(0,signals.shape[2]-1, num=num)]:
                optimizer.zero_grad()
                predictions = model(signals[:, :, :t+1])

                reconstruction_loss = loss_criterion(predictions, labels[:,t].to(device))
                epoch_loss += reconstruction_loss.item()
                reconstruction_loss.backward()
                optimizer.step()

        test_loss = test_model_rt_rg(model,valid_loader)
        train_loss_trend.append(epoch_loss/(num*(i+1)))
        test_loss_trend.append(test_loss)

        if epoch % 10 ==0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', epoch_loss/(num*(i+1)))
            print('Test ===>loss: ', test_loss)

    test_loss = test_model_rt_rg(model,valid_loader)
    print('Test loss: ', test_loss)

    # Save model and results
    if not os.path.exists(os.path.join("./ckpt/",data)):
        os.mkdir(os.path.join("./ckpt/",data))
    torch.save(model.state_dict(), './ckpt/' + data + '/' + str(experiment) + '.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join('./plots',data,'train_loss.png'))


def test_model_rt(model,test_loader,num=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    correct_label_test = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    test_loss = 0
    for i, (signals,labels) in enumerate(test_loader):
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
        for t in [int(tt) for tt in np.linspace(0,signals.shape[2]-2,num=num)]:
        #for t in [24]:
            prediction = model(signals[:,:,:t+1])
            predicted_label = (prediction > 0.5).float()
            labels_th = (labels[:,t] > 0.5).float()
            auc, recall, precision, correct = evaluate(labels_th.contiguous().view(-1), predicted_label.contiguous().view(-1), prediction.contiguous().view(-1))
            correct_label_test += correct
            auc_test += auc
            recall_test +=  recall
            precision_test +=  precision
            count +=  1
            loss = torch.nn.BCELoss()(prediction, labels[:,t].to(device))
            test_loss += loss.item()

    test_loss = test_loss/((i+1)*num)
    return test_loss, recall_test, precision_test, auc_test/((i+1)*num), correct_label_test


def test_model_rt_rg(model,test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    num=50
    for i, (signals,labels) in enumerate(test_loader):
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
        for t in [int(tt) for tt in np.linspace(0,signals.shape[2]-1,num=num)]:
            prediction = model(signals[:,:,:t+1])
            loss = torch.nn.MSELoss()(prediction, labels[:,t].to(device))
            test_loss += loss.item()

    test_loss = test_loss/(num*(i+1))
    return test_loss


def train_reconstruction(model, train_loader, valid_loader, n_epochs, device, experiment):
    train_loss_trend = []
    test_loss_trend = []
    model.to(device)

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.0001, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0.9, weight_decay=1e-4)

    for epoch in range(n_epochs + 1):
        model.train()
        epoch_loss = 0
        for i, (signals, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            signals, _ = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
            mu, logvar, z = model.encode(signals)
            recon = model.decode(z)
            loss = torch.nn.MSELoss()(recon, signals[:,:,-1].view(len(signals),-1)) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            epoch_loss = + loss.item()
            loss.backward()
            optimizer.step()
        test_loss = test_reconstruction(model, valid_loader, device)

        train_loss_trend.append(epoch_loss)
        test_loss_trend.append(test_loss)
        if epoch % 10 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', epoch_loss)
            print('Test ===>loss: ', test_loss)

    # Save model and results
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    torch.save(model.state_dict(), './ckpt/' + str(experiment) + '.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig('train_loss.pdf')


def test_reconstruction(model, valid_loader, device):
    model.eval()
    test_loss = 0
    for i, (signals, labels) in enumerate(valid_loader):
        signals, _ = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
        mu, logvar, z = model.encode(signals)
        recon = model.decode(z)
        loss = torch.nn.MSELoss()(recon, signals[:,:,-1].view(len(signals),-1)) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        test_loss = + loss.item()
    return test_loss


def load_data(batch_size, path='./data/', **kwargs):
    p_data = PatientData(path)

    features = kwargs['features'] if 'features' in kwargs.keys() else range(p_data.train_data.shape[1])
    p_data.train_data = p_data.train_data[:,features,:]
    p_data.test_data = p_data.test_data[:,features,:]
    p_data.feature_size = len(features)
    n_train = int(0.8*p_data.n_train)
    if 'cv' in kwargs.keys():
        kf = KFold(n_splits=5,random_state=42)
        #print(p_data.train_data[:,:,0].shape,kf.split(p_data.train_data))
        train_idx, valid_idx = list(kf.split(p_data.train_data))[kwargs['cv']]
    else:
        train_idx = range(n_train)
        valid_idx = range(n_train, p_data.n_train)

    train_dataset = utils.TensorDataset(torch.Tensor(p_data.train_data[train_idx, :, :]),
                                        torch.Tensor(p_data.train_label[train_idx]))
    valid_dataset = utils.TensorDataset(torch.Tensor(p_data.train_data[valid_idx, :, :]),
                                        torch.Tensor(p_data.train_label[valid_idx]))
    test_dataset = utils.TensorDataset(torch.Tensor(p_data.test_data), torch.Tensor(p_data.test_label))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=p_data.n_train - int(0.8 * p_data.n_train))
    test_loader = DataLoader(test_dataset, batch_size=p_data.n_test)
    print('Train set: ', np.count_nonzero(p_data.train_label[0:int(0.8 * p_data.n_train)]),
          'patient who died out of %d total'%(int(0.8 * p_data.n_train)),
          '(Average missing in train: %.2f)' % (np.mean(p_data.train_missing[0:int(0.8 * p_data.n_train)])))
    print('Valid set: ', np.count_nonzero(p_data.train_label[int(0.8 * p_data.n_train):]),
          'patient who died out of %d total'%(len(p_data.train_label[int(0.8 * p_data.n_train):])),
          '(Average missing in validation: %.2f)' % (np.mean(p_data.train_missing[int(0.8 * p_data.n_train):])))
    print('Test set: ', np.count_nonzero(p_data.test_label), 'patient who died  out of %d total'%(len(p_data.test_data)),
          '(Average missing in test: %.2f)' % (np.mean(p_data.test_missing)))
    return p_data, train_loader, valid_loader, test_loader


def load_ghg_data(batch_size, path='./data_generator/data',**kwargs):
    p_data = GHGData(path,transform=None) #data already normalized zero mean 1 std
    #print('ghg label stats', np.mean(p_data.train_label),np.std(p_data.train_label))
    features=kwargs['features'] if 'features' in kwargs.keys() else range(p_data.train_data.shape[1]) 
    p_data.train_data = p_data.train_data[:,features,:]
    p_data.test_data  = p_data.test_data[:,features,:]

    n_train = int(0.8 * len(x_train))
    if 'cv' in kwargs.keys():
        kf = KFold(n_splits=5, random_state=42)
        train_idx,valid_idx = list(kf.split(x_train))[kwargs['cv']]
    else:
        train_idx = range(n_train)
        valid_idx = range(ntrain.len(x_train))

    train_dataset = utils.TensorDataset(torch.Tensor(p_data.train_data[train_idx, :, :]),
                                        torch.Tensor(p_data.train_label[train_idx]))
    valid_dataset = utils.TensorDataset(torch.Tensor(p_data.train_data[train_idx, :, :]),
                                        torch.Tensor(p_data.train_label[train_idx]))
    test_dataset = utils.TensorDataset(torch.Tensor(p_data.test_data[:,:,:]), torch.Tensor(p_data.test_label))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=p_data.n_train - int(0.8 * p_data.n_train))
    test_loader = DataLoader(test_dataset, batch_size=p_data.n_test)
    #print('Train set: ', p_data.train_data.shape)
    #print('Valid set: ', p_data.train_data.shape)
    #print('Test set: ', p_data.test_data.shape)
    p_data.feature_size = len(features)
    return p_data, train_loader, valid_loader, test_loader


def load_simulated_data(batch_size=100, path='./data/simulated_data', data_type='state', percentage=1., **kwargs):
    if data_type=='state':
        with open('./data/simulated_data/state_dataset_importance_train.pkl', 'rb') as f:
            importance_score_train = pkl.load(f)
        with open('./data/simulated_data/state_dataset_importance_test.pkl', 'rb') as f:
            importance_score_train_test = pkl.load(f)
        file_name = 'state_dataset_'
    else:
        file_name = ''
    with open(os.path.join(path, file_name+'x_train.pkl'), 'rb') as f:
        x_train = pkl.load(f)
    with open(os.path.join(path, file_name+'y_train.pkl'), 'rb') as f:
        y_train = pkl.load(f)
    with open(os.path.join(path, file_name+'x_test.pkl'), 'rb') as f:
        x_test = pkl.load(f)
    with open(os.path.join(path, file_name+'y_test.pkl'), 'rb') as f:
        y_test = pkl.load(f)

    features = kwargs['features'] if 'features' in kwargs.keys() else list(range(x_test.shape[1]))
    
    total_sample_n = int(len(x_train)*percentage)
    x_train = x_train[:total_sample_n]
    y_train = y_train[:total_sample_n]
    n_train = int(0.8 * len(x_train))
    x_train = x_train[:,features,:]
    x_test = x_test[:,features,:]

    n_train = int(0.8 * len(x_train))
    if 'cv' in kwargs.keys():
        kf = KFold(n_splits=5, random_state=42)
        train_idx,valid_idx = list(kf.split(x_train))[kwargs['cv']]
    else:
        train_idx = range(n_train)
        valid_idx = range(n_train,len(x_train))
    
    train_dataset = utils.TensorDataset(torch.Tensor(x_train[train_idx, :, :]),
                                        torch.Tensor(y_train[train_idx, :]))
    valid_dataset = utils.TensorDataset(torch.Tensor(x_train[valid_idx, :, :]),
                                        torch.Tensor(y_train[valid_idx, :]))
    test_dataset = utils.TensorDataset(torch.Tensor(x_test[:,:,:]), torch.Tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=len(x_train) - int(0.8 * n_train))
    test_loader = DataLoader(test_dataset, batch_size=len(x_test))
    return np.concatenate([x_train, x_test]), train_loader, valid_loader, test_loader


def logistic(x):
    return 1./(1+np.exp(-1*x))


def top_risk_change(exp):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    span = []
    testset = list(exp.test_loader.dataset)
    for i,(signal, label) in enumerate(testset):
        exp.risk_predictor.load_state_dict(torch.load('./ckpt/mimic/risk_predictor.pt'))
        exp.risk_predictor.to(device)
        exp.risk_predictor.eval()
        risk = []
        for t in range(1,48):
            risk.append(exp.risk_predictor(signal[:, 0:t].view(1, signal.shape[0], t).to(device)).item())
        span.append((i, max(risk) - min(risk)))
    span.sort(key=lambda pair:pair[1], reverse=True)
    print([x[0] for x in span[0:300]])


def test_cond(mean, covariance, sig_ind, x_ind):
    x_ind = x_ind.unsqueeze(-1)
    mean_1 = torch.cat((mean[:, :sig_ind], mean[:, sig_ind + 1:]), 1).unsqueeze(-1)
    cov_1_2 = torch.cat(([covariance[:, 0:sig_ind, sig_ind], covariance[:, sig_ind + 1:, sig_ind]]), 1).unsqueeze(-1)
    cov_2_2 = covariance[:, sig_ind, sig_ind]
    cov_1_1 = torch.cat(([covariance[:, 0:sig_ind, :], covariance[:, sig_ind + 1:, :]]), 1)
    cov_1_1 = torch.cat(([cov_1_1[:, :, 0:sig_ind], cov_1_1[:, :, sig_ind + 1:]]), 2)
    mean_cond = mean_1 + torch.bmm(cov_1_2, (x_ind - mean[:, sig_ind]).unsqueeze(-1)) / cov_2_2
    covariance_cond = cov_1_1 - torch.bmm(cov_1_2, torch.transpose(cov_1_2, 2, 1)) / cov_2_2
    return mean_cond, covariance_cond


def shade_state(gt_importance_subj, t, ax, data):
    # Shade the state on simulation data plots
    if gt_importance_subj.shape[0] == 3:
        gt_importance_subj = gt_importance_subj.transpose(1, 0)
    if not data == 'simulation_spike':
        prev_color = 'g' if np.argmax(gt_importance_subj[:, 1]) < np.argmax(gt_importance_subj[:, 2]) else 'y'
        for ttt in range(1, len(t) + 1):
            if gt_importance_subj[ttt, 1] == 1:
                ax.axvspan(ttt - 1, ttt, facecolor='g', alpha=0.3)
                prev_color = 'g'
            elif gt_importance_subj[ttt, 2] == 1:
                ax.axvspan(ttt - 1, ttt, facecolor='y', alpha=0.3)
                prev_color = 'y'
            elif not prev_color is None:
                ax.axvspan(ttt - 1, ttt, facecolor=prev_color, alpha=0.3)
    else:
        for ttt in range(1, len(t) + 1):
            if gt_importance_subj[ttt] == 1:
                ax.axvspan(ttt - 1, ttt, facecolor='g', alpha=0.3)


def plot_importance(subject, signals, label, a, a_std, a_max, n_feats_to_plot, signals_to_analyze, color_map, fmap, data, gt_importance_subj, save_path, patient_data):
    important_signals = []
    markers = ['*', 'D', 'X', 'o', '8', 'v', '+']
    f, axs = plt.subplots(6)
    t = np.arange(1, a[0].shape[-1]+1)

    # a[0] = 2./(1.+np.exp(-3*a[0])) - 1.
    if hasattr(patient_data, 'test_intervention'):
        f_color = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
        for int_ind, intervention in enumerate(patient_data.test_intervention[subject, :, :]):
            if sum(intervention) != 0:
                switch_point = []
                intervention = intervention[1:]
                for i in range(1, len(intervention)):
                    if intervention[i] != intervention[i - 1]:
                        switch_point.append(i)
                if len(switch_point) % 2 == 1:
                    switch_point.append(len(intervention) - 1)
                for count in range(int(len(switch_point) / 2)):
                    if count == 0:
                        axs[0].axvspan(xmin=switch_point[count * 2], xmax=switch_point[2 * count + 1],
                                    facecolor=f_color[int_ind % len(f_color)], alpha=0.2,
                                    label='%s' % (intervention_list[int_ind]))
                    else:
                        axs[0].axvspan(xmin=switch_point[count * 2], xmax=switch_point[2 * count + 1],
                                    facecolor=f_color[int_ind % len(f_color)], alpha=0.2)

    if not gt_importance_subj is None:
        shade_state(gt_importance_subj, t, axs[0], data)

    for ax_ind, ax in enumerate(axs[1:]):
        ax.grid()
        ax.tick_params(axis='both', labelsize=36)
        ax.set_ylabel('importance', fontweight='bold', fontsize=32)
        # if ax_ind in [0, 1, 2]:
        #     ax.set_ylim(bottom=-0.02, top=1.)

        for ind, sig in a_max[ax_ind][0:n_feats_to_plot]:
            ind = int(ind)
            ref_ind = signals_to_analyze[ind]
            if ref_ind not in important_signals:
                important_signals.append(ref_ind)
            c = color_map[ref_ind]
            if np.sum(a_std[ax_ind][ind])>0:
                ax.errorbar(t, a[ax_ind][ind, :], yerr=a_std[ax_ind][ind, :], marker=markers[list(important_signals).index(ref_ind)%len(markers)],
                            linewidth=3, elinewidth=1, markersize=9, markeredgecolor='k', color=c, label='%s' % (fmap[ref_ind]))
            else:
                ax.errorbar(t, a[ax_ind][ind, :], yerr=a_std[ax_ind][ind, :], linewidth=3, elinewidth=1, color=c, label='%s'%(fmap[ref_ind]))
    important_signals = np.unique(important_signals)
    max_plot = (torch.max(torch.abs(signals[important_signals, :]))).item()
    for i, ref_ind in enumerate(important_signals):
        c = color_map[ref_ind]
        if data == 'mimic':
            axs[0].plot(np.array(signals[ref_ind, 1:]) / max_plot, linewidth=3, color=c, label='%s' % (fmap[ref_ind]))
        else:
            axs[0].plot(np.array(signals[ref_ind, 1:]), linewidth=3, color=c, label='%s' % (fmap[ref_ind]))
    axs[0].tick_params(axis='both', labelsize=36)
    axs[0].grid()

    axs[0].plot(np.array(label), '--', linewidth=6, label='Risk score', c='black')
    axs[0].set_title('Time series signals and Model\'s predicted risk', fontweight='bold', fontsize=40)
    axs[1].set_title('Feature importance FIT', fontweight='bold', fontsize=40)
    axs[2].set_title('Feature importance AFO', fontweight='bold', fontsize=40)
    axs[3].set_title('Feature importance FO', fontweight='bold', fontsize=40)
    axs[4].set_title('Feature importance Sensitivity analysis', fontweight='bold', fontsize=40)
    axs[5].set_title('Feature importance Attention', fontweight='bold', fontsize=40)
    axs[5].set_xlabel('time', fontweight='bold', fontsize=32)
    axs[0].set_ylabel('signal value', fontweight='bold', fontsize=32)

    f.set_figheight(40)
    f.set_figwidth(60)
    plt.subplots_adjust(hspace=.5)
    plt.savefig(os.path.join(save_path, 'feature_%d.pdf' % (subject)), dpi=300, orientation='landscape')
    fig_legend = plt.figure(figsize=(13, 1.2))
    handles, labels = axs[0].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper left', ncol=4, fancybox=True, handlelength=6, fontsize='xx-large')
    fig_legend.savefig(os.path.join(save_path, 'legend_%d.pdf' %subject), dpi=300, bbox_inches='tight')

    for imp_plot_ind in range(4):
        heatmap_fig = plt.figure(figsize=(15, 1) if data == 'simulation' else (16, 9))
        plt.yticks(rotation=0)
        imp_plot = sns.heatmap(a[imp_plot_ind], yticklabels=fmap,
                               square=True if data == 'mimic' else False)  # , vmin=0, vmax=1)
        heatmap_fig.savefig(os.path.join(save_path, 'heatmap_%s_%s.pdf'%(str(subject), ['FIT', 'AFO', 'FO', 'Sens'][imp_plot_ind])))
    if data == 'simulation':
        heatmap_gt = plt.figure(figsize=(20, 1))
        plt.yticks(rotation=0)
        imp_plot = sns.heatmap(gt_importance_subj, yticklabels=fmap)
        heatmap_gt.savefig(os.path.join(save_path,'heatmap_%s_ground_truth.pdf'%str(subject)))

    return important_signals
