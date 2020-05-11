import numpy as np
import torch
import os
import sys
import re

# from TSX.generator import JointFeatureGenerator, train_joint_feature_generator, JointDistributionGenerator
from TSX.utils import load_simulated_data, AverageMeter

from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tnrange, tqdm_notebook
import matplotlib.pyplot as plt

from TSX.generator import train_joint_feature_generator, JointDistributionGenerator
from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency
#import shap
import lime
import lime.lime_tabular


# class DistGenerator:
#     def __init__(self, model, train_loader, n_componenets=5):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.base_model = model
#         self.base_model.device = self.device
#         self.distribution = JointDistributionGenerator(n_componenets, train_loader)
#
#     def attribute(self, x, y, retrospective=False):
#         x = x.to(self.device)
#         _, n_features, t_len = x.shape
#         score = np.zeros(x.shape)
#         if retrospective:
#             p_y_t = self.base_model(x)
#
#         for t in range(1, t_len):
#             if not retrospective:
#                 p_y_t = self.base_model(x[:, :, :t + 1])
#             for i in range(n_features):
#                 x_hat = x[:, :, 0:t + 1].clone()
#                 kl_all = []
#                 for _ in range(10):
#                     conditional = self.distribution.forward_conditional(x[:, :, t], [i])
#                     # print('sample', x[2, :, t])
#                     # print('Cond', conditional[2])
#                     x_hat[:, :, t] = conditional.to(self.device)
#                     y_hat_t = self.base_model(x_hat)
#                     # kl = torch.nn.KLDivLoss(reduction='none')(torch.Tensor(np.log(y_hat_t)).to(self.device), p_y_t)
#                     kl = torch.abs(y_hat_t[:, :] - p_y_t[:, :])
#                     # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
#                     kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
#                 E_kl = np.mean(np.array(kl_all), axis=0)
#                 # score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
#                 score[:, i, t] = E_kl
#         return score


class FITExplainer:
    def __init__(self, model, generator=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = generator
        self.base_model = model.to(self.device)

    def fit_generator(self, generator_model, train_loader, test_loader, n_epochs=300):
        train_joint_feature_generator(generator_model, train_loader, test_loader, generator_type='joint_generator',
                                      n_epochs=500, lr=0.001, weight_decay=0)
        self.generator = generator_model.to(self.device)

    def attribute(self, x, y, n_samples=10, retrospective=False, distance_metric='kl'):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        self.generator.eval()
        self.generator.to(self.device)
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = self.base_model(x)

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.base_model(x[:, :, :t+1])
            for i in range(n_features):
                x_hat = x[:,:,0:t+1].clone()
                div_all=[]
                p_tm1 = self.base_model(x[:,:,0:t])
                for _ in range(n_samples):
                    x_hat_t, _ = self.generator.forward_conditional(x[:, :, :t], x[:, :, t], [i])
                    x_hat[:, :, t] = x_hat_t
                    y_hat_t = self.base_model(x_hat)
                    if distance_metric == 'kl':
                        # kl = torch.nn.KLDivLoss(reduction='none')(torch.Tensor(np.log(y_hat_t)).to(self.device), p_y_t)
                        div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_tm1), p_y_t), -1) - \
                             torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t), -1)
                        # print(x_hat_t[0], x[0, :, t], self.generator.forward_joint(x[:, :, :t])[0])
                        # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                        div_all.append(div.cpu().detach().numpy())
                    elif distance_metric == 'mean_divergence':
                        div = torch.abs(y_hat_t - p_y_t)
                        div_all.append(np.mean(div.detach().cpu().numpy(), -1))
                E_div = np.mean(np.array(div_all), axis=0)
                if distance_metric=='kl':
                    # print('********** E_div')
                    score[:, i, t] = 2./(1+np.exp(-1*E_div)) - 1
                elif distance_metric=='mean_divergence':
                    score[:, i, t] = 1-E_div
        return score


class FFCExplainer:
    def __init__(self, model, generator=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = generator
        self.base_model = model.to(self.device)

    def fit_generator(self, generator_model, train_loader, test_loader, n_epochs=300):
        train_joint_feature_generator(generator_model, train_loader, test_loader, generator_type='joint_generator', n_epochs=n_epochs)
        self.generator = generator_model.to(self.device)

    def attribute(self, x, y, n_samples=10, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        self.generator.eval()
        self.generator.to(self.device)
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = self.base_model(x)

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.base_model(x[:, :, :min((t+1), t_len)])
            for i in range(n_features):
                x_hat = x[:,:,0:t+1].clone()
                kl_all=[]
                for _ in range(n_samples):
                    x_hat_t = self.generator.forward_joint(x[:, :, :t])
                    x_hat[:, i, t] = x_hat_t[:,i]
                    y_hat_t = self.base_model(x_hat)
                    kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    kl_all.append(torch.sum(kl, -1).detach().cpu().numpy())
                E_kl = np.mean(np.array(kl_all),axis=0)
                score[:,i,t] =E_kl #* 1e-6
        return score


class FOExplainer:
    def __init__(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = model.to(self.device)

    def attribute(self, x, y, retrospective=False,n_samples=10):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = self.base_model(x)

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.base_model(x[:, :, :t+1])
            for i in range(n_features):
                x_hat = x[:,:,0:t+1].clone()
                kl_all=[]
                for _ in range(n_samples):
                    x_hat[:, i, t] = torch.Tensor(np.random.uniform(-3,+3, size=(len(x),)))#torch.Tensor(np.array([np.random.uniform(-3,+3)]).reshape(-1)).to(self.device)
                    y_hat_t = self.base_model(x_hat)
                    # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    kl = torch.abs(y_hat_t-p_y_t)
                    # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all),axis=0)
                # score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
                score[:, i, t] = E_kl
        return score


class AFOExplainer:
    def __init__(self, model, train_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = model.to(self.device)
        trainset = list(train_loader.dataset)
        self.data_distribution = torch.stack([x[0] for x in trainset])

    def attribute(self, x, y, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = self.base_model(x)

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.base_model(x[:, :, :t + 1])
            for i in range(n_features):
                feature_dist = (np.array(self.data_distribution[:, i, :]).reshape(-1))
                x_hat = x[:,:,0:t+1].clone()
                kl_all=[]
                for _ in range(10):
                    x_hat[:, i, t] = torch.Tensor(np.random.choice(feature_dist, size=(len(x),))).to(self.device)
                    y_hat_t = self.base_model(x_hat)
                    kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    # kl = torch.abs(y_hat_t[:, :] - p_y_t[:, :])
                    kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                    # kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
                # score[:, i, t] = E_kl
        return score


class RETAINexplainer:
    def __init__(self, model, data):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.data = data

    def _epoch(self, loader, criterion, optimizer=None, train=False):
        if train and not optimizer:
            raise AttributeError("Optimizer should be given for training")

        if train:
            self.model.train()
            mode = 'Train'
        else:
            self.model.eval()
            mode = 'Eval'

        losses = AverageMeter()
        labels = []
        outputs = []

        for bi, batch in enumerate(tqdm_notebook(loader, desc="{} batches".format(mode), leave=False)):
            inputs, targets = batch
            lengths = torch.randint(low=10, high=inputs.shape[2], size=(len(inputs),))
            lengths, _ = torch.sort(lengths, descending=True)
            lengths[0] = inputs.shape[-1]
            inputs = inputs.permute(0, 2, 1)  # Shape: (batch, length, features)
            targets = targets[torch.range(0, len(inputs) - 1).long(), lengths - 1]

            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(targets)
            input_var = input_var.to(self.device)
            target_var = target_var.to(self.device)

            output, alpha, beta = self.model(input_var, lengths)
            loss = criterion(output, target_var.long())
            # print(loss.data[0])
            # assert not np.isnan(loss.data[0]), 'Model diverged with loss = NaN'

            labels.append(targets)

            # since the outputs are logit, not probabilities
            outputs.append(torch.nn.functional.softmax(output).data)

            # record loss
            losses.update(loss.item(), inputs.size(0))

            # compute gradient and do update step
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg

    def fit_model(self, train_loader, valid_loader, test_loader, epochs=10, lr=0.001, plot=False):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.95)

        best_valid_epoch = 0
        best_valid_loss = sys.float_info.max
        best_valid_auc = 0.0
        best_valid_aupr = 0.0

        train_losses = []
        valid_losses = []

        if plot:
            # initialise the graph and settings
            fig = plt.figure(figsize=(12, 9))  # , facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)
            plt.ion()
            fig.show()
            fig.canvas.draw()

        for ei in tnrange(epochs, desc="Epochs"):
            # Train
            train_y_true, train_y_pred, train_loss = self._epoch(train_loader, criterion=criterion,
                                                                 optimizer=optimizer,
                                                                 train=True)
            train_losses.append(train_loss)

            # Eval
            valid_y_true, valid_y_pred, valid_loss = self._epoch(valid_loader, criterion=criterion)
            valid_losses.append(valid_loss)

            print("Epoch {} - Loss train: {}, valid: {}".format(ei, train_loss, valid_loss))

            valid_y_true.to(self.device)
            valid_y_pred.to(self.device)

            valid_auc = roc_auc_score(valid_y_true.cpu().numpy(), valid_y_pred.cpu().numpy()[:, 1], average="weighted")
            valid_aupr = average_precision_score(valid_y_true.cpu().numpy(), valid_y_pred.cpu().numpy()[:, 1],
                                                 average="weighted")

            is_best = valid_auc > best_valid_auc

            if is_best:
                best_valid_epoch = ei
                best_valid_loss = valid_loss
                best_valid_auc = valid_auc
                best_valid_aupr = valid_aupr

                # print("\t New best validation AUC!")
                # print('\t Evaluation on the test set')

                # evaluate on the test set
                test_y_true, test_y_pred, test_loss = self._epoch(test_loader, criterion=criterion)

                train_y_true.to(self.device)
                train_y_pred.to(self.device)
                test_y_true.to(self.device)
                test_y_pred.to(self.device)

                train_auc = roc_auc_score(train_y_true.cpu().numpy(), train_y_pred.cpu().numpy()[:, 1],
                                          average="weighted")
                train_aupr = average_precision_score(train_y_true.cpu().numpy(), train_y_pred.cpu().numpy()[:, 1],
                                                     average="weighted")

                test_auc = roc_auc_score(test_y_true.cpu().numpy(), test_y_pred.cpu().numpy()[:, 1], average="weighted")
                test_aupr = average_precision_score(test_y_true.cpu().numpy(), test_y_pred.cpu().numpy()[:, 1],
                                                    average="weighted")

                # print("Train - Loss: {}, AUC: {}".format(train_loss, train_auc))
                # print("Valid - Loss: {}, AUC: {}".format(valid_loss, valid_auc))
                # print(" Test - Loss: {}, AUC: {}".format(valid_loss, test_auc))

                with open('./outputs/retain_train_result.txt', 'w') as f:
                    f.write('Best Validation Epoch: {}\n'.format(ei))
                    f.write('Best Validation Loss: {}\n'.format(best_valid_loss))
                    f.write('Best Validation AUROC: {}\n'.format(best_valid_auc))
                    f.write('Best Validation AUPR: {}\n'.format(best_valid_aupr))
                    f.write('Train Loss: {}\n'.format(train_loss))
                    f.write('Train AUROC: {}\n'.format(train_auc))
                    f.write('Train AUPR: {}\n'.format(train_aupr))
                    f.write('Test Loss: {}\n'.format(test_loss))
                    f.write('Test AUROC: {}\n'.format(test_auc))
                    f.write('Test AUPR: {}\n'.format(test_aupr))

                if not os.path.exists("./ckpt/%s" % self.data):
                    os.mkdir("./ckpt/%s" % self.data)
                torch.save(self.model.state_dict(), './ckpt/%s/retain.pt' % self.data)

            # plot
            if plot:
                ax.clear()
                ax.plot(np.arange(len(train_losses)), np.array(train_losses), label='Training Loss')
                ax.plot(np.arange(len(valid_losses)), np.array(valid_losses), label='Validation Loss')
                ax.set_xlabel('epoch')
                ax.set_ylabel('Loss')
                ax.legend(loc="best")
                plt.tight_layout()
                plt.savefig(os.path.join('./plots', self.data, 'retain_train_loss.pdf'))
                # fig.canvas.draw()

        print('Best Validation Epoch: {}\n'.format(best_valid_epoch))
        print('Best Validation Loss: {}\n'.format(best_valid_loss))
        print('Best Validation AUROC: {}\n'.format(best_valid_auc))
        print('Best Validation AUPR: {}\n'.format(best_valid_aupr))
        print('Test Loss: {}\n'.format(test_loss))
        print('Test AUROC: {}\n'.format(test_auc))
        print('Test AUPR: {}\n'.format(test_aupr))

    def attribute(self, x, y):
        score = np.zeros(x.shape)
        x = x.permute(0, 2, 1)  # shape:[batch, time, feature]
        logit, alpha, beta = self.model(x, (torch.ones((len(x),)) * x.shape[1]).long())
        w_emb = self.model.embedding[1].weight
        for i in range(x.shape[2]):
            for t in range(x.shape[1]):
                imp = self.model.output(beta[:, t, :] * w_emb[:, i].expand_as(beta[:, t, :]))
                score[:, i, t] = (alpha[:, t, 0] * imp[torch.range(0, len(imp) - 1).long(), y.long()] * x[:, t,
                                                                                                        i]).detach().cpu().numpy()
        return score


class DeepLiftExplainer:
    def __init__(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = model.to(self.device)
        self.explainer = DeepLift(self.base_model)

    def attribute(self, x, y, retrospective=False):
        print(x.shape, y.shape)
        self.base_model.zero_grad()
        if retrospective:
            score = self.explainer.attribute(x, target=y.long(), baselines=(x * 0))
            score = score.detach().cpu().numpy()
        else:
            score = np.zeros(x.shape)
            for t in range(1,x.shape[-1]):
                imp = self.explainer.attribute(x[:,:,:t+1], target=y.long(), baselines=(x[:,:,:t+1] * 0))
                score[:, :, t] = imp.detach().cpu().numpy()[:,:,-1]
        return score


class IGExplainer:
    def __init__(self, model):
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = model.to(self.device)
        self.base_model.device=self.device
        self.explainer = IntegratedGradients(self.base_model)

    def attribute(self, x, y, retrospective=False):
        x, y = x.to(self.device), y.to(self.device)
        self.base_model.zero_grad()
        if retrospective:
            score = self.explainer.attribute(x, target=y.long(), baselines=(x * 0))
            score = score.detach().cpu().numpy()
        else:
            score = np.zeros(x.shape)
            for t in range(1,x.shape[-1]):
                imp = self.explainer.attribute(x[:,:,:t+1], target=y.long(), baselines=torch.zeros(x[:,:,:t+1].shape).to(self.device))
                score[:, :, t] = imp.detach().cpu().numpy()[:,:,-1]
        return score


class GradientShapExplainer:
    def __init__(self, model):
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = model.to(self.device)
        self.base_model.device=self.device
        self.explainer = GradientShap(self.base_model)

    def attribute(self, x, y, retrospective=False):
        x, y = x.to(self.device), y.to(self.device)
        if retrospective:
            score = self.explainer.attribute(x, target=y[:, -1].long(),
                                             n_samples=50, stdevs=0.0001, baselines=torch.cat([x * 0, x * 1]))
            score = score.cpu().numpy()
        else:
            score = np.zeros(x.shape)
            for t in range(1,x.shape[-1]):
                imp = self.explainer.attribute(x[:,:,:t+1], target=y[:, t].long(),
                                             n_samples=50, stdevs=0.0001, baselines=torch.cat([x[:,:,:t+1] * 0, x[:,:,:t+1] * 1]))
                score[:, :, t] = imp.cpu().numpy()[:,:,-1]
        return score


class SHAPExplainer:
    def __init__(self, model, train_loader):
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = model
        self.base_model.device=self.device
        trainset = list(train_loader.dataset)
        x_train = torch.stack([x[0] for x in trainset])
        background = x_train[np.random.choice(np.arange(len(x_train)), 100, replace=False)]
        self.explainer = shap.DeepExplainer(self.base_model, background.to(self.device))

    def attribute(self, x, y, retrospective=False):
        x.to(self.device)
        if retrospective:
            score = self.explainere.shap_values(x)
        else:
            score = np.zeros(x.shape)
            for t in range(1,x.shape[-1]):
                imp = self.explainer.shap_values(x[:,:,:t+1].reshape((len(x), -1)))
                score[:, :, t] = imp.detach().cpu().numpy()[:,:,-1]
        return score


class LIMExplainer:
    def __init__(self, model, train_loader):
        self.device = 'cpu' #if torch.cuda.is_available() else 'cpu'
        self.base_model = model
        self.base_model.device = self.device
        trainset = list(train_loader.dataset)
        x_train = torch.stack([x[0] for x in trainset]).to(self.device)
        x_train = x_train[torch.arange(len(x_train)), :, torch.randint(10,x_train.shape[-1], (len(x_train),))]
        self.explainer = lime.lime_tabular.LimeTabularExplainer(x_train.cpu().numpy(), feature_names = ['f%d'%c for c in range(x_train.shape[1])],
                                                                discretize_continuous=True)

    def _predictor_wrapper(self, sample):
        """
        In order to use the lime explainer library we need to go back and forth between numpy library (compatible with Lime)
        and torch (Compatible with the predictor model). This wrapper helps with this
        :param sample: input sample for the predictor (type: numpy array)
        :return: one-hot model output (type: numpy array)
        """
        torch_in = torch.Tensor(sample).reshape(len(sample),-1,1)
        torch_in.to(self.device)
        out = self.base_model(torch_in)
        return out.detach().cpu().numpy()

    def attribute(self, x, y, retrospective=False):
        x = x.cpu().numpy()
        score = np.zeros(x.shape)
        p_y_t_vec = np.zeros((x.shape[0], x.shape[-1]))
        for t in range(1, x.shape[-1]):
            p_y_t = self.base_model(x[:, :, :t + 1])
            p_y_t_vec[:, t - 1] = np.array([p > 0.5 for p in p_y_t.cpu().detach().numpy()[:, 1]]).flatten()
        for sample_ind, sample in enumerate(x):
            for t in range(1, x.shape[-1]):
                imp = self.explainer.explain_instance(sample[:, t], self._predictor_wrapper, top_labels=x.shape[1])
                for ind, st in enumerate(imp.as_list()):
                    imp_score = st[1]
                    terms = re.split('< | > | <= | >=', st[0])
                    for feat in range(x.shape[1]):
                        if 'f%d'%feat in terms:
                            score[sample_ind, feat, t] = imp_score
        return score, p_y_t_vec

