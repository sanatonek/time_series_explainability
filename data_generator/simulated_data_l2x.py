import numpy as np
import pickle
import argparse
import os
from scipy.signal import butter, lfilter, freqz
import timesynth as ts

np.random.seed(42)

SIG_NUM = 10
STATE_NUM = 1
P_S0 = [0.5]

imp_feature = [[1, 2, 3, 4], [5, 6, 7, 8]]  # Features that are always set as important
trans_mat = np.array([[0.1, 0.9], [0.1, 0.9]])
correlated_feature = {0: {1: [6, 7], 3: [4]}, 1: {5: [2, 3, 4], 8: [5, 7]}}

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

order = 6
fs = 30.0       # sample rate, Hz
cutoff = 6.8 # desired cutoff frequency of the filter, Hz


def next_state(previous_state, t):
    # params = [(abs(p-0.1)+timing_factor)/2. for p in previous_state]
    # print(params,previous_state)
    # params = [abs(p - 0.1) for p in previous_state]
    # print(previous_state)
    # params = [abs(p) for p in trans_mat[int(previous_state),1-int(previous_state)]]
    # params = trans_mat[int(previous_state),1-int(previous_state)]
    if previous_state == 1:
        params = 0.95
    else:
        params = 0.05
    # params = 0.2
    # print('previous', previous_state)
    params = params - float(t / 500) if params > 0.8 else params
    # print('transition probability',params)
    next_st = np.random.binomial(1, params)
    return next_st


def init_distribution_params():
    # Covariance matrix is constant across states but distribution means change based on the state value
    state_count = np.power(2, STATE_NUM)
    cov = np.eye(SIG_NUM) * 0.8
    covariance = []
    for i in range(state_count):
        cc = cov.copy()
        for k, v in correlated_feature[i].items():
            cc[k, v] = 0.01
            for vv in v:
                cc[vv, k] = 0.01
        # print(cc)
        cc = cc + np.eye(SIG_NUM) * 1e-3
        covariance.append(cc)
    covariance = np.array(covariance)
    mean = []
    for i in range(state_count):
        m = np.zeros(SIG_NUM)
        mean.append(m)
        # print(m)
    mean = np.array(mean)
    return mean, covariance


def state_decoder(previous, next_st):
    return int(next_st * (1 - previous) + (1 - next_st) * previous)


def generate_XOR_labels(X):
    y = np.exp(X[:, 0] * X[:, 1])

    prob_1 = np.expand_dims(1 / (1 + y), 1)
    prob_0 = np.expand_dims(y / (1 + y), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:, :4] ** 2, axis=1) - 4.0)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_additive_labels(X):
    logit = np.exp(-10 * np.sin(-0.2 * X[:, 0]) + abs(X[:, 1]) + X[:, 2] + np.exp(-X[:, 3]) - 2.4)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def create_signal(sig_len, gp_params):
    signal = []
    state_local = []
    y = []
    importance = []
    y_logits = []

    previous = np.random.binomial(1, P_S0)[0]
    previous_label = None
    delta_state = 0
    sample = np.array([gp.sample_vectorized(time_vector=np.array(range(sig_len))) for gp in gp_params])
    #sample = np.array([butter_lowpass_filter(sample[f,:], cutoff, fs, order) for f in range(SIG_NUM)])
    #print(sample)
    for ii in range(sig_len):
        next_st = next_state(previous, delta_state)
        state_n = next_st

        if state_n == previous:
            delta_state += 1
        else:
            delta_state = 0

        sample[-1, ii] = 0.5 * (1 - state_n) + -0.5 * state_n
        #importance.append(imp_sig)
        # sample_ii = np.random.multivariate_normal(np.zeros(SIG_NUM), np.eye(SIG_NUM))
        #sample_ii = np.random.multivariate_normal(mean[state_n], cov[state_n])
        #sample_ii[-1] += 3 * (1 - state_n) + -3 * state_n
        previous = state_n
        #signal.append(sample_ii)

        sample_ii = (sample[:,ii]).reshape((1, -1))
        y_probs = state_n * generate_additive_labels(sample_ii[:, imp_feature[state_n]]) + \
                  (1 - state_n) * generate_orange_labels(sample_ii[:, imp_feature[state_n]])
        y_logit = y_probs[0][1]
        y_label = np.random.binomial(1, y_logit)

        imp_sig = np.zeros(SIG_NUM)
        #if y_label != previous_label or ii == 0:
        imp_sig[imp_feature[state_n]] = 1
        imp_sig[-1] = 1
        importance.append(imp_sig)

        previous_label = y_label
        #print(y_label)

        # print('previous state:', previous, 'next state probability:', next_st, 'delta_state:', delta_state,
        #      'current state:', state_n, 'ylogit', y_logit)
        # print('sample', sample_ii)

        y.append(y_label)
        y_logits.append(y_logit)
        state_local.append(state_n)
    signal = sample
    #print(signal.shape)
    y = np.array(y)
    importance = np.array(importance)
    #print(importance.shape)
    return signal, y, state_local, importance, y_logits


def decay(x):
    return [0.9 * (1 - 0.1) ** x, 0.9 * (1 - 0.1) ** x]


def logit(x):
    return 1. / (1 + np.exp(-2 * (x)))


def normalize(train_data, test_data, config='mean_normalized'):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_size = train_data.shape[1]
    len_of_stay = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)
    if config == 'mean_normalized':
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide='ignore', invalid='ignore')
        train_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in train_data])
        test_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in test_data])
    elif config == 'zero_to_one':
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])
    return train_data_n, test_data_n


def create_dataset(count, signal_len):
    dataset = []
    labels = []
    importance_score = []
    states = []
    label_logits = []
    # mean, cov = init_distribution_params()
    gp_lengthscale = np.random.uniform(0.5,2.5, SIG_NUM)
    means = [1.2, 0.8,1.5, 0.4, 0.5, -1.2, -1.5, -0.8, -0.4, -0.5]
    gp_vec = [ts.signals.GaussianProcess(lengthscale=g, mean=m, variance=0.5) for g,m in zip(gp_lengthscale,means)]
    for num in range(count):
        sig, y, state, importance, y_logits = create_signal(signal_len, gp_params=gp_vec)  # , mean, cov)
        dataset.append(sig)
        labels.append(y)
        importance_score.append(importance.T)
        states.append(state)
        label_logits.append(y_logits)

        if num %50==0:
            print(num, count)
    dataset = np.array(dataset)
    labels = np.array(labels)
    importance_score = np.array(importance_score)
    states = np.array(states)
    label_logits = np.array(label_logits)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    #train_data_n, test_data_n = normalize(train_data, test_data)
    train_data_n = train_data
    test_data_n = test_data
    if not os.path.exists('./data/simulated_data_l2x'):
        os.mkdir('./data/simulated_data_l2x')
    with open('./data/simulated_data_l2x/state_dataset_x_train.pkl', 'wb') as f:
        pickle.dump(train_data_n, f)
    with open('./data/simulated_data_l2x/state_dataset_x_test.pkl', 'wb') as f:
        pickle.dump(test_data_n, f)
    with open('./data/simulated_data_l2x/state_dataset_y_train.pkl', 'wb') as f:
        pickle.dump(labels[:n_train], f)
    with open('./data/simulated_data_l2x/state_dataset_y_test.pkl', 'wb') as f:
        pickle.dump(labels[n_train:], f)
    with open('./data/simulated_data_l2x/state_dataset_importance_train.pkl', 'wb') as f:
        pickle.dump(importance_score[:n_train], f)
    with open('./data/simulated_data_l2x/state_dataset_importance_test.pkl', 'wb') as f:
        pickle.dump(importance_score[n_train:], f)
    with open('./data/simulated_data_l2x/state_dataset_logits_train.pkl', 'wb') as f:
        pickle.dump(label_logits[:n_train], f)
    with open('./data/simulated_data_l2x/state_dataset_logits_test.pkl', 'wb') as f:
        pickle.dump(label_logits[n_train:], f)
    with open('./data/simulated_data_l2x/state_dataset_states_train.pkl', 'wb') as f:
        pickle.dump(states[:n_train], f)
    with open('./data/simulated_data_l2x/state_dataset_states_test.pkl', 'wb') as f:
        pickle.dump(states[n_train:], f)

    return dataset, labels, states,label_logits


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_len', type=int, default=100, help='Length of the signal to generate')
    parser.add_argument('--signal_num', type=int, default=1000, help='Number of the signals to generate')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    np.random.seed(234)
    dataset, labels, states,label_logits = create_dataset(args.signal_num, args.signal_len)

    if args.plot:
        import matplotlib.pyplot as plt

        f, (x1, x2) = plt.subplots(2, 1)
        state_1_1 = []
        state_1_0 = []
        state_0_1 = []
        state_0_0 = []

        idx0 = np.where(states == 0)
        idx1 = np.where(states == 1)

        for c in range(len(idx0[0])):
            if labels[idx0[0][c], idx0[1][c]] == 0:
                state_0_0.append(labels[idx0[0][c], idx0[1][c]])
            else:
                state_0_1.append(labels[idx0[0][c], idx0[1][c]])

        for c in range(len(idx1[0])):
            if labels[idx1[0][c], idx1[1][c]] == 0:
                state_1_0.append(labels[idx1[0][c], idx1[1][c]])
            else:
                state_1_1.append(labels[idx1[0][c], idx1[1][c]])
        x1.hist(state_0_0,label='label 0')
        x1.hist(state_0_1, label = 'label 1')
        x1.set_title('state 0: orange')
        x1.legend()
        
        x2.hist(state_1_0,label='label 0')
        x2.hist(state_1_1,label = 'label 1')
        x2.set_title('state 1: additive')
        x2.legend()
        plt.savefig('plot.pdf')


        f, (x1,x2) = plt.subplots(2,1)
        for id in range(len(labels)):
            for i, sample in enumerate(dataset[id]):
                if labels[id,i]:
                    x1.scatter(sample[0], sample[1], c='r')
                else:
                    x1.scatter(sample[0], sample[1], c='b')
                if states[id,i]:
                    x2.scatter(sample[0], sample[1], c='b')
                else:
                    x2.scatter(sample[0], sample[1], c='r')
            x1.set_title('Distribution based on label')
            x2.set_title('Distribution based on state')
        plt.savefig('plot2.pdf')

        plot_id=2
        f= plt.figure(figsize=(18,9))
        x1 = f.subplots()
        for i in range(SIG_NUM):
            x1.plot(range(dataset.shape[2]), dataset[plot_id, i, :], linewidth=1, label='feature %d' % (0))
        #x1.plot(range(dataset.shape[2]), dataset[plot_id, 1, :], linewidth=3, label='feature %d' % (1))
        #x1.plot(range(dataset.shape[2]), dataset[plot_id, 2, :], linewidth=3, label='feature %d' % (2))
        x1.plot(range(dataset.shape[2]), label_logits[plot_id, :], linewidth=3, label='label')
        plt.legend()
        plt.savefig('plotsample_l2x.pdf')
