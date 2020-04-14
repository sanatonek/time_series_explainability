import numpy as np
import pickle
import argparse
import os

SIG_NUM = 10
STATE_NUM = 1
P_S0 = [0.5]

imp_feature = [[1, 2, 3, 4], [5, 6, 7, 8]]  # Features that are always set as important
trans_mat = np.array([[0.1, 0.9], [0.1, 0.9]])


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
    logit = np.exp(-100 * np.sin(0.2 * X[:, 0]) + abs(X[:, 1]) + X[:, 2] + np.exp(-X[:, 3]) - 2.4)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def create_signal(sig_len):
    signal = []
    state_local = []
    y = []
    importance = []
    y_logits = []

    previous = np.random.binomial(1, P_S0)[0]
    delta_state = 0
    for ii in range(sig_len):
        next_st = next_state(previous, delta_state)
        state_n = next_st

        if state_n == previous:
            delta_state += 1
        else:
            delta_state = 0

        imp_sig = np.zeros(SIG_NUM)
        if state_n != previous or ii == 0:
            imp_sig[imp_feature[state_n]] = 1

        importance.append(imp_sig)
        sample_ii = np.random.multivariate_normal(np.zeros(SIG_NUM), np.eye(SIG_NUM))
        sample_ii[-1] += 3 * (1 - state_n) + -3 * state_n
        previous = state_n
        signal.append(sample_ii)

        sample_ii = sample_ii.reshape((1, -1))
        y_probs = state_n * generate_additive_labels(sample_ii[:, imp_feature[state_n]] - 0.3) + \
                  (1 - state_n) * generate_orange_labels(sample_ii[:, imp_feature[state_n]] + 0.3)
        y_logit = y_probs[0][1]
        y_label = np.random.binomial(1, y_logit)

        # print('previous state:', previous, 'next state probability:', next_st, 'delta_state:', delta_state,
        #      'current state:', state_n, 'ylogit', y_logit)
        # print('sample', sample_ii)

        y.append(y_label)
        y_logits.append(y_logit)
        state_local.append(state_n)
    signal = np.array(signal)
    y = np.array(y)
    importance = np.array(importance)
    # print(importance.shape)
    return signal.T, y, state_local, importance, y_logits


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
    for num in range(count):
        sig, y, state, importance, y_logits = create_signal(signal_len)  # , mean, cov)
        dataset.append(sig)
        labels.append(y)
        importance_score.append(importance.T)
        states.append(state)
        label_logits.append(y_logits)
    dataset = np.array(dataset)
    labels = np.array(labels)
    importance_score = np.array(importance_score)
    states = np.array(states)
    label_logits = np.array(label_logits)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    # train_data_n, test_data_n = normalize(train_data, test_data)
    train_data_n = train_data
    test_data_n = test_data
    if not os.path.exists('./data/simulated_data'):
        os.mkdir('./data/simulated_data')
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

    return dataset, labels, states


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_len', type=int, default=10, help='Length of the signal to generate')
    parser.add_argument('--signal_num', type=int, default=100, help='Number of the signals to generate')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    np.random.seed(234)
    dataset, labels, states = create_dataset(args.signal_num, args.signal_len)

    if args.plot:
        import matplotlib.pyplot as plt

        f, (x1, x2) = plt.subplots(2, 1)
        state_1_1 = []
        state_1_0 = []
        state_0_1 = []
        state_0_0 = []

        idx0 = np.where(states == 0)
        idx1 = np.where(states == 1)

        print(len(idx0[0]))
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
        x1.hist(state_0_0)
        x1.hist(state_0_1)
        x2.hist(state_1_0)
        x2.hist(state_1_1)
        plt.savefig('plot.pdf')