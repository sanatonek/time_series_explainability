import numpy as np
from scipy.stats import multivariate_normal as mn

def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    #print(observations.shape)
    # forward part of the algorithm
    fwd = []
    f_prev = {}
    for i in range(observations.shape[1]):
        observation_i = observations[:,i]
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)

            f_curr[st] = mn.pdf(observation_i, mean=emm_prob[st]['mean'], cov=emm_prob[st]['cov']) * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    
    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)
    #print(p_fwd)

    # backward part of the algorithm
    bkw = []
    b_prev = {}
    #for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
    for i in range(observations.shape[1]):
        if i==0:
            observation_i_plus = None
        else:
            observation_i_plus = observations[:,-i]
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * mn.pdf(observation_i_plus, mean=emm_prob[l]['mean'],cov=emm_prob[l]['cov']) * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * mn.pdf(observations[:,0],emm_prob[l]['mean'],emm_prob[l]['cov'])* b_curr[l] for l in states)

    # merging the two parts
    posterior = []
    for i in range(observations.shape[1]):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    #print(p_fwd,p_bkw)
    #assert p_fwd == p_bkw #all numerical errors
    return p_fwd, p_bkw, posterior

'''
def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    # forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # backward part of the algorithm
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    assert p_fwd == p_bkw
    return fwd, bkw, posterior
'''