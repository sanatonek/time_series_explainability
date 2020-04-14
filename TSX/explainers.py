import numpy as np
import torch
import os

from TSX.generator import JointFeatureGenerator, train_joint_feature_generator
from TSX.utils import load_simulated_data
from TSX.models import StateClassifier


class FITExplainer:
    def __init__(self, model, generator=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = generator
        self.base_model = model.to(self.device)

    def fit_generator(self, generator_model, train_loader, test_loader):
        train_joint_feature_generator(generator_model, train_loader, test_loader, generator_type='joint', n_epochs=100)
        self.generator = generator_model.to(self.device)

    def score(self, x, n_samples=10):
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
        p_y_T = self.base_model(x)

        for t in range(1, t_len):
            for i in range(n_features):
                y_hat_all = []
                x_hat = x[:,:,0:t+1].clone()
                for _ in range(n_samples):
                    x_hat_t, _ = self.generator.forward_conditional(x[:, :, :t], x[:, :, t], [i])
                    x_hat[:, :, t] = x_hat_t
                    y_hat_t = self.base_model(x_hat).detach().cpu().numpy()
                    y_hat_all.append(y_hat_t)

                # KL divergence
                y_hat_mean = np.mean(y_hat_all, axis=0)
                y_hat_std = np.std(y_hat_all, axis=0)

                kl = torch.nn.KLDivLoss(reduction='none')(torch.Tensor(np.log(y_hat_mean)).to(self.device), p_y_T) ## Figure out where the expectation should be
                kl = torch.sum(kl, -1) # TODO: check this!!
                score[:,i,t] = 1./(kl.cpu().detach().numpy()+1e-6) * 1e-6
        return score


if __name__=='__main__':
    # Prepare a model to explain
    _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100,path='./data/simulated_data', percentage=0.8)
    model = StateClassifier(feature_size=3, n_state=2, hidden_size=100)
    model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt'%('simulation', 'joint_encoder'))))

    # Set up the explainer
    fit = FITExplainer(model, train_loader, valid_loader)
    generator = JointFeatureGenerator(3, hidden_size=10, data='simulation')
    fit.fit_generator(generator, train_loader, valid_loader)

    for x,y in test_loader:
        score = fit.score(x)