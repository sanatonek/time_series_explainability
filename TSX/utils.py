import numpy as np
import os
import torch
import torch.utils.data as utils
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from TSX.models import PatientData, NormalPatientData
import matplotlib.pyplot as plt

# Ignore sklearn warnings caused by ill-defined precision score (caused by single class prediction)
import warnings
warnings.filterwarnings("ignore")


def evaluate(labels, predicted_label, predicted_probability):
    labels_array = np.array(labels.cpu())
    prediction_prob_array = np.array(predicted_probability.view(len(labels), -1).detach().cpu())
    prediction_array = np.array(predicted_label.cpu())
    auc = roc_auc_score(np.array(labels.cpu()), np.array(predicted_probability.view(len(labels), -1).detach().cpu()))
    # recall = recall_score(labels_array, prediction_array)
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
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
        # out = model(x.mean(dim=2).reshape((len(y),-1)))
        out = model(x)
        prediction = (out > 0.5).view(len(y), ).float()
        auc, recall, precision, correct = evaluate(y, prediction, out)
        correct_label += correct
        auc_test = + auc
        recall_test = + recall
        precision_test = + precision
        count = + 1
        loss = + criteria(out.view(len(y), ), y).item()
        total += len(x)
    return recall_test, precision_test, auc_test / count, correct_label, loss


def train(train_loader, model, device, optimizer, loss_criterion=torch.nn.BCELoss(), verbose=True):
    model = model.to(device)
    model.train()
    recall_train, precision_train, auc_train, correct_label, epoch_loss = 0, 0, 0, 0, 0
    count = 0
    for i, (signals, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
        # loss_criterion = torch.nn.BCELoss(weight=8*labels+1)
        risks = model(signals)
        predicted_label = (risks > 0.5).view(len(labels), ).float()

        auc, recall, precision, correct = evaluate(labels, predicted_label, risks)
        correct_label += correct
        auc_train = + auc
        recall_train = + recall
        precision_train = + precision
        count = + 1

        loss = loss_criterion(risks.view(len(labels), ), labels)
        epoch_loss = + loss.item()
        loss.backward()
        optimizer.step()
    return recall_train, precision_train, auc_train / count, correct_label, epoch_loss


def train_model(model, train_loader, valid_loader, optimizer, n_epochs, device, experiment):
    train_loss_trend = []
    test_loss_trend = []

    for epoch in range(n_epochs + 1):
        recall_train, precision_train, auc_train, correct_label_train, epoch_loss = train(train_loader, model,
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
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    torch.save(model.state_dict(), './ckpt/' + str(experiment) + '.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig('train_loss.png')


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
    plt.savefig('train_loss.png')


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


def load_data(batch_size, path='./data_generator/data', *argv):
    if 'generator_data' in argv:
        p_data = NormalPatientData(path)
    else:
        p_data = PatientData(path)
    train_dataset = utils.TensorDataset(torch.Tensor(p_data.train_data[0:int(0.8 * p_data.n_train), :, :]),
                                        torch.Tensor(p_data.train_label[0:int(0.8 * p_data.n_train)]))
    valid_dataset = utils.TensorDataset(torch.Tensor(p_data.train_data[int(0.8 * p_data.n_train):, :, :]),
                                        torch.Tensor(p_data.train_label[int(0.8 * p_data.n_train):]))
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
