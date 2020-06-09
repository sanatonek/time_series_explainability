import numpy as np
from data_generator.hmm_forward import *
import data_generator.state_data as sd #use parameters from here.
import data_generator.simulated_l2x_switchstate as l2x
import pickle as pkl
from sklearn.mixture import GaussianMixture

class TrueFeatureGenerator():

    def __init__(self, data):
        if data=='simulation':
            params = sd
        elif data=='simulation_l2x':
            params = l2x
        self.data=data

        self.states = list(range(params.STATE_NUM))
        self.start_probability={}
        for s in self.states:
            self.start_probability[s] = params.P_S0[0]
 
        self.transition_probability = {}
        for s in self.states:
            self.transition_probability[s] = {}
            for ss in self.states:
                self.transition_probability[s][ss] = params.trans_mat[s,ss]

        self.mean, self.cov = params.init_distribution_params()

        self.emission_probability = {}
        for s in self.states:
            self.emission_probability[s]={}
            self.emission_probability[s]['mean'] = self.mean[s]
            self.emission_probability[s]['cov'] = self.cov[s]


        
    def sample(self,x,t,feature):
        observations = x[:,:t]
        p_s_past={} #p(s_t-1|X_{0:t-1})
        for st in self.states:
            p_s_past[st],_,_ = fwd_bkw(observations, self.states, self.start_probability, self.transition_probability, self.emission_probability,st)

        #p_currstate_past={'Healthy': 0., 'Sick':0.} #p(s_t | X_{0:t-1})
        p_currstate_past={}
        for s in self.states:
            p_currstate_past[s] = 0.

        for curr_state in self.states: 
            for st in self.states:
                p_currstate_past[curr_state] += self.transition_probability[curr_state][st]*p_s_past[st]
        
        gmm = GaussianMixture(n_components = len(self.states),covariance_type='full')
        gmm.fit(np.random.randn(10, observations.shape[0]))
        gmm.weights_ = list(p_currstate_past.values())
        gmm.means_ = np.array(self.mean)
        gmm.covariances_ = np.array(self.cov)
        for i in  range(2):
            gmm.precisions_[i] = np.linalg.inv(gmm.covariances_[i])
            gmm.precisions_cholesky_[i] = np.linalg.cholesky(gmm.covariances_[i])

        x_sampled = gmm.sample()[0]
        return x_sampled[0,feature]

    def log_prob(self,x,t,feature,samples):
        observations = x[:,:t]
        p_s_past={} #p(s_t-1|X_{0:t-1})
        for st in self.states:
            p_s_past[st],_,_ = fwd_bkw(observations, self.states, self.start_probability, self.transition_probability, self.emission_probability,st)

        p_currstate_past={}
        for s in self.states:
            p_currstate_past[s] = 0.

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

    
