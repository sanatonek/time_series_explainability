import numpy as np
from data_generator.hmm_forward import *
import data_generator.state_data as sd #use parameters from here.
import pickle as pkl
from sklearn.mixture import GaussianMixture

with open('./data/simulated_data/state_dataset_importance_train.pkl','rb') as f:
    state_data_train = pkl.load(f)
    
with open('./data/simulated_data/state_dataset_importance_train.pkl','rb') as f:
    state_data_test = pkl.load(f)
    
with open('./data/simulated_data/state_dataset_x_train.pkl','rb') as f:
    x_train = pkl.load(f)
    
with open('./data/simulated_data/state_dataset_x_test.pkl','rb') as f:
    x_test = pkl.load(f)

with open('./data/simulated_data/state_dataset_y_train.pkl','rb') as f:
    y_train = pkl.load(f)
    
with open('./data/simulated_data/state_dataset_y_test.pkl','rb') as f:
    y_test = pkl.load(f)

class TrueFeatureGenerator():

    def __init__(self):
        self.states = ('Healthy', 'Sick')
        self.start_probability = {'Healthy': sd.P_S0[0], 'Sick': sd.P_S0[0]}
 
        self.transition_probability = {
        'Healthy' : {'Healthy': sd.trans_mat[0,0], 'Sick': sd.trans_mat[0,1]},
         'Sick' : {'Healthy': sd.trans_mat[1,0], 'Sick': sd.trans_mat[1,1]},
        }

        self.mean, self.cov = sd.init_distribution_params()

        self.emission_probability = {
       'Healthy' : {'mean': self.mean[0], 'cov': self.cov[0]},
       'Sick' : {'mean': self.mean[1], 'cov': self.cov[1]},
        }
        
    def sample(self,x,t,feature):
        observations = x[:,:t]
        p_s_past={} #p(s_t-1|X_{0:t-1})
        for st in self.states:
            p_s_past[st],_,_ = fwd_bkw(observations, self.states, self.start_probability, self.transition_probability, self.emission_probability,st)

        p_currstate_past={'Healthy': 0., 'Sick':0.} #p(s_t | X_{0:t-1})

        for curr_state in self.states: 
            for st in self.states:
                p_currstate_past[curr_state] += self.transition_probability[curr_state][st]*p_s_past[st]
        
        gmm = GaussianMixture(n_components = len(self.states),covariance_type='full')
        gmm.fit(np.random.randn(10, observations.shape[0]))
        gmm.weights_ = list(p_currstate_past.values())
        gmm.means_ = np.array(self.mean)
        gmm.covariances_ = np.array(self.cov)
        for i in range(2):
            gmm.precisions_[i] = np.linalg.inv(gmm.covariances_[i])
            gmm.precisions_cholesky_[i] = np.linalg.cholesky(gmm.covariances_[i])
        
        x_sampled = gmm.sample()[0]
        return x_sampled[0,feature]

    def log_prob(self,x,t,feature,samples):
        observations = x[:,:t]
        p_s_past={} #p(s_t-1|X_{0:t-1})
        for st in self.states:
            p_s_past[st],_,_ = fwd_bkw(observations, self.states, self.start_probability, self.transition_probability, self.emission_probability,st)

        p_currstate_past={'Healthy': 0., 'Sick':0.} #p(s_t | X_{0:t-1})

        for curr_state in self.states: 
            for st in self.states:
                p_currstate_past[curr_state] += self.transition_probability[curr_state][st]*p_s_past[st]
        
        gmm = GaussianMixture(n_components = len(self.states),covariance_type='full')
        gmm.fit(np.random.randn(10, observations.shape[0]))
        gmm.weights_ = list(p_currstate_past.values())
        gmm.means_ = np.array(self.mean)
        gmm.covariances_ = np.array(self.cov)
        for i in range(2):
            gmm.precisions_[i] = np.linalg.inv(gmm.covariances_[i])
            gmm.precisions_cholesky_[i] = np.linalg.cholesky(gmm.covariances_[i])
        
        return gmm.score_samples(samples)

    
