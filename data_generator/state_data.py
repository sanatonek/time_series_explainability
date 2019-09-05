import numpy as np
import pickle
import argparse
import os

SIG_NUM = 3
STATE_NUM = 1
P_S0 = [0.5]

correlated_feature = [2, 1] # Features that re correlated with the important feature in each state
imp_feature = 0  # Feature that is always set as important
scale = [30, 10]  # Scaling factor for distribution mean in each state


def init_distribution_params():
    # Covariance matrix is constant across states but distribution means change based on the state value
    state_count = np.power(2,STATE_NUM)
    corr = abs(np.random.randn(SIG_NUM))*20
    cov = np.diag(corr)
    covariance = []
    for i in range(state_count):
        c = cov.copy()
        c[imp_feature,correlated_feature[i]] = 10
        c[correlated_feature[i], imp_feature] = 10
        c = c + np.eye(SIG_NUM)*1e-3
        covariance.append(c)
    covariance = np.array(covariance)
    mean = []
    for i in range(state_count):
        m = (np.random.randn(SIG_NUM)+1.)*scale[i]
        mean.append(m)
    mean = np.array(mean)
    return mean, covariance


def next_state(previous_state, t):
    timing_factor = 1./(1+np.exp(-t/100))
    params = [(abs(p-0.1)+timing_factor)/2. for p in previous_state]
    #params = [abs(p - 0.2) for p in previous_state]
    next = np.random.binomial(1, params)
    return next


def state_decoder(state_one_hot):
    base = 1
    state = 0
    for digit in state_one_hot:
        state = state + base*digit
        base = base * 2
    return state


def create_signal(sig_len):
    mean, cov = init_distribution_params()
    signal = []
    states = []
    y = []
    importance = []

    previous = np.random.binomial(1, P_S0)
    for i in range(sig_len):
        next = next_state(previous, i)
        state_n = state_decoder(next)
        imp_sig = [1, 0, 0]
        imp_sig[correlated_feature[state_n]] = 1
        importance.append(np.zeros(SIG_NUM) if previous==next else imp_sig)
        sample = np.random.multivariate_normal(mean[state_n], cov[state_n])
        previous = next
        signal.append(sample)
        y.append(np.random.binomial(1, logit(sample)))
        states.append(state_n)
    signal = np.array(signal)
    y = np.array(y)
    importance = np.array(importance)
    return signal.T, y, states, importance


def logit(x):
    return 1./(1+np.exp(-1*x[imp_feature]))


def create_dataset(count, signal_len):
    dataset = []
    labels = []
    importance_score = []
    states = []
    for num in range(count):
        sig, y, state, importance = create_signal(signal_len)
        dataset.append(sig)
        labels.append(y)
        importance_score.append(importance)
        states.append(state)
    dataset = np.array(dataset)
    labels = np.array(labels)
    importance_score = np.array(importance_score)
    states = np.array(states)
    n_train= int(len(dataset)*0.8)
    if not os.path.exists('./data/simulated_data'):
        os.mkdir('./data/simulated_data')
    with open('./data/simulated_data/state_dataset_x_train.pkl', 'wb') as f:
        pickle.dump(dataset[:n_train], f)
    with open('./data/simulated_data/state_dataset_x_test.pkl', 'wb') as f:
        pickle.dump(dataset[n_train:], f)
    with open('./data/simulated_data/state_dataset_y_train.pkl', 'wb') as f:
        pickle.dump(labels[:n_train], f)
    with open('./data/simulated_data/state_dataset_y_test.pkl', 'wb') as f:
        pickle.dump(labels[n_train:], f)
    with open('./data/simulated_data/state_dataset_importance_train.pkl', 'wb') as f:
        pickle.dump(importance_score[:n_train], f)
    with open('./data/simulated_data/state_dataset_importance_test.pkl', 'wb') as f:
        pickle.dump(importance_score[n_train:], f)
    return dataset, labels, states


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_len', type=int, default=200, help='Length of the signal to generate')
    parser.add_argument('--signal_num', type=int, default=1000, help='Number of the signals to generate')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    np.random.seed(234)
    dataset, labels, states = create_dataset(args.signal_num, args.signal_len)

    if args.plot:
        import matplotlib.pyplot as plt
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
        plt.show()

