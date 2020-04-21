import numpy as np
import os
import torch
from torch import nn
import random
import torch.utils.data as utils
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Ignore sklearn warnings caused by ill-defined precision score (caused by single class prediction)
import warnings
warnings.filterwarnings("ignore")


def evaluate(labels, predicted_label, predicted_probability):
    labels_array = labels.detach().cpu().numpy()
    prediction_array = predicted_label.detach().cpu().numpy()
    l_idx = []
    for l in range(labels_array.shape[1]):
        if len(np.unique(labels_array[:,l]))>=2:
            l_idx.append(l)
    auc = roc_auc_score(labels_array[:,l_idx], np.array(predicted_probability[:,l_idx].detach().cpu()))
    report = classification_report(labels_array, prediction_array,labels=list(range(labels_array.shape[1])),output_dict=True)
    recall=0
    precision=0
    correct_label=0
    recall = report['macro avg']['recall']
    precision =report['macro avg']['precision']
    correct_label = np.equal(np.argmax(labels_array,1), np.argmax(prediction_array,1)).sum()
    return auc, recall, precision, correct_label


def test(test_loader, model, device, criteria=torch.nn.CrossEntropyLoss(), verbose=True):
    model.to(device)
    correct_label = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    total = 0
    auc_test = 0
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
        pred_onehot = torch.FloatTensor(y.shape[0], y.shape[1])
        logits = model(x)
        m = nn.Softmax(dim=1)
        out = m(logits)
        _, predicted_label = out.max(1)
        pred_onehot.zero_()
        pred_onehot.scatter_(1, predicted_label.view(-1,1), 1)
        auc, recall, precision, correct = evaluate(y, pred_onehot, out)
        correct_label += correct
        auc_test = auc_test + auc
        recall_test = + recall
        precision_test = + precision
        count = + 1
        loss = + criteria(out, torch.max(y, 1)[1]).item()
        total += len(x)
    return recall_test, precision_test, auc_test/(i+1), correct_label, loss


def create_new_labels(test_loader, model, device, verbose=True):
    model.to(device)
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
        pred_onehot = torch.FloatTensor(y.shape[0], y.shape[1])
        logits = model(x)
        m = nn.Softmax(dim=1)
        out = m(logits)
        _, predicted_label = out.max(1)
        pred_onehot.zero_()
        pred_onehot.scatter_(1, predicted_label.view(-1,1), 1)
        
    return pred_onehot.detach().cpu().numpy(), logits.detach().cpu().numpy()


def train(train_loader, model, device, optimizer, loss_criterion=torch.nn.CrossEntropyLoss()):
    model = model.to(device)
    model.train()
    auc_train = 0
    recall_train, precision_train, auc_train, correct_label, epoch_loss = 0, 0, 0, 0, 0
    
    for i, (signals, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        pred_onehot = torch.FloatTensor(labels.shape[0], labels.shape[1])
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
        logits = model(signals)
        m = nn.Softmax(dim=1)
        probabilities = m(logits)
        _, predicted_label = probabilities.max(1)
        pred_onehot.zero_()
        pred_onehot.scatter_(1, predicted_label.view(-1,1), 1)
        auc, recall, precision, correct = evaluate(labels, pred_onehot, probabilities)
        correct_label += correct
        auc_train = auc_train + auc
        recall_train = + recall
        precision_train = + precision
        loss = loss_criterion(probabilities, torch.max(labels, 1)[1])
        epoch_loss = + loss.item()
        loss.backward()
        optimizer.step()
    return recall_train, precision_train, auc_train/(i+1), correct_label, epoch_loss, i+1


def train_model(model, train_loader, valid_loader, optimizer, n_epochs, device, experiment,data='ddg'):
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
    

def load_simulated_data(x_train, y_train, x_test, y_test, batch_size=100, percentage=1., **kwargs):

    features = kwargs['features'] if 'features' in kwargs.keys() else list(range(x_test.shape[1]))
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    
    scaler = StandardScaler()
    x_train_flat = scaler.fit_transform(np.reshape(x_train,[x_train.shape[0],-1]))
    x_train = np.reshape(x_train_flat,x_train.shape)
    x_test_flat = scaler.transform(np.reshape(x_test,[x_test.shape[0],-1]))
    x_test = np.reshape(x_test_flat,x_test.shape)

    n_train = int(0.8 * x_train.shape[0])
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, )
    valid_loader = DataLoader(valid_dataset, batch_size=len(x_train) - int(0.8 * n_train))
    test_loader = DataLoader(test_dataset, batch_size=len(x_test))
    return np.concatenate([x_train, x_test]), train_loader, valid_loader, test_loader



class EncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, regres = True, rnn="GRU", bidirectional=False, return_all=False,
                 seed=random.seed('2019'),data='mimic',n_classes=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)

        self.regressor = nn.Sequential(nn.BatchNorm1d(self.hidden_size),
                                       nn.Tanh(),
                                       nn.Dropout(0.5),
                                       nn.Linear(self.hidden_size,self.hidden_size),
                                       nn.Tanh(),
                                       nn.Dropout(0.5),
                                       nn.Linear(self.hidden_size,self.hidden_size),
                                       nn.Tanh(),
                                       nn.Dropout(0.5),
                                       nn.Linear(self.hidden_size, n_classes))

    def forward(self, ip, past_state=None):
        ip = ip.permute(2, 0, 1).to(self.device)
        if not past_state:
            past_state = torch.zeros([1, ip.shape[1], self.hidden_size]).to(self.device)
        if self.rnn_type == 'GRU':
            all_encodings, encoding = self.rnn(ip, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(ip, (past_state, past_state))
        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                reshaped_encodings = all_encodings.view(all_encodings.shape[1]*all_encodings.shape[0],-1)
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0],-1))
        else:
            return encoding.view(encoding.shape[1], -1)

class Experiment(ABC):
    def __init__(self, train_loader, valid_loader, test_loader, data='mimic'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.data = data
        if not os.path.exists('./ckpt'):
            os.mkdir('./ckpt')
        self.ckpt_path ='./ckpt/' + self.data
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)


    @abstractmethod
    def run(self):
        raise RuntimeError('Function not implemented')

    def train(self, n_epochs, learn_rt=False):
        raise RuntimeError('Function not implemented')
        
        
class EncoderPredictor(Experiment):
    """ Baseline mortality prediction using an encoder to encode patient status, and a risk predictor to predict risk of mortality
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor',simulation=False,data='mimic', model='RNN',n_classes=1):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader, data=data)
        if model=='RNN':
            self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=False,data=data,n_classes = n_classes)
        self.model_type = model
        self.experiment = experiment
        self.data = data
        self.n_classes = n_classes

    def run(self, train,n_epochs, **kwargs):
        if train:
            self.train(n_epochs=n_epochs)
        else:
            path = './ckpt/' + self.data + '/' + str(self.experiment) + '_' + self.model_type + '.pt'
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path))
                new_labels_train,logits_train = create_new_labels(self.train_loader, self.model, self.device)
                new_labels_test,logits_test = create_new_labels(self.test_loader, self.model, self.device)
                _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            else:
                raise RuntimeError('No saved checkpoint for this model')
            return new_labels_train, logits_train, new_labels_test, logits_test

    def train(self, n_epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-3)
        train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                            self.experiment+'_'+self.model_type, data=self.data)
        # Evaluate performance on held-out test set
        _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
        print('\nFinal performance on held out test set ===> AUC: ', auc_test)

