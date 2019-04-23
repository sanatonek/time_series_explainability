import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
import pickle as pkl
import os


def main(n_samples, plot):
    signal_in = []
    #signal_out = []
    thresholds = []
    for i in range(n_samples):
        x, y, t = generate_sample(plot)
        signal_in.append(x)
        #signal_out.append(y)
        thresholds.append(t)
    signal_in = np.array(signal_in)
    #signal_out = np.array(signal_out)
    return signal_in, thresholds
        

def generate_sample(plot):
    seed = np.random.randint(1,100)
    # Correlation coefficients for x2=f(x1)
    coeff = [0.4,0.5,0.01]
    noise = ts.noise.GaussianNoise(std=0.3)
    trend_style = {0:'increase', 1:'decrease', 2:'hill', 3:'valley'}
    trend = np.random.randint(4)

    x1 = ts.signals.NARMA(seed=seed)
    x1_ts = ts.TimeSeries(x1, noise_generator=noise)
    x1_sample, signals, errors = x1_ts.sample(np.array(range(48)))
    #x1_sample = x1.sample_vectorized(np.array(range(48)))

    x2_sample = coeff[0]*x1_sample + coeff[1]*x1_sample*x1_sample + coeff[2]*x1_sample*x1_sample*x1_sample
    if trend==0 or trend==1:
        t = np.random.randint(20,38)
        x2_sample[t:] = x2_sample[t:] + (1 if trend==0 else -1) * np.log(0.3*np.asarray(range(48-t))+1.) 
    elif trend==2 or trend==3:
        t = np.sort(np.random.choice(np.arange(15,38),4))
        s = +1 if trend==2 else -1
        x2_sample[t[0]:t[1]] = x2_sample[t[0]:t[1]] + s*np.log(0.5*np.asarray(range(t[1]-t[0]))+1.)
        x2_sample[t[1]:t[2]] = x2_sample[t[1]:t[2]] + s * np.log(0.5 * np.full(x2_sample[t[1]:t[2]].shape, (t[1]-t[0])) + 1.)
        x2_sample[t[2]:t[3]] = x2_sample[t[2]:t[3]] - s*np.log(0.5*np.asarray(range(t[3]-t[2]))+1.)

    x2_sample += .5*np.random.randn(len(x2_sample))

    x3 = ts.signals.NARMA(10,[.5,.2,2.5,.5], seed=seed*5000)
    x3_ts = ts.TimeSeries(x3, noise_generator=noise)
    x3_sample, signals, errors = x3_ts.sample(np.array(range(48)))
    #x3_sample = x3.sample_vectorized(np.array(range(48)))

    y = logistic(0.5*x1_sample*x1_sample + 0.5*x2_sample*x2_sample + 0.5*x3_sample*x3_sample)

    if plot:
        plt.plot(x1_sample, label='x1')
        plt.plot(x2_sample, label='x2')
        plt.plot(x3_sample, label='x3')
        plt.plot(y)
        plt.title('Sample style: %s'%(trend_style[trend]))
        if isinstance(t,np.ndarray):
            for thresh in t:
                plt.axvline(x=thresh, color='grey')
        else:
            plt.axvline(x=t, color='grey')
        plt.legend()
        plt.show()

    return np.stack([x1_sample, x2_sample, x3_sample]), y, t


def save_data(path,array):
    with open(path,'wb') as f:
        pkl.dump(array, f)


def logistic(x):
    return 1./(1+np.exp(-1*x))


if __name__=='__main__':
    n_samples = 20000
    signal_in, thresholds = main(n_samples=n_samples, plot=False)
    n_train = int(0.8*n_samples)
    x_train = signal_in[0:n_train,:,:]
    thresholds_train = thresholds[0:n_train]
    x_test = signal_in[n_train:,:,:]
    thresholds_test = thresholds[n_train:]

    feature_min = np.tile(np.min(np.min(x_train, axis=0), axis=-1), (48, 1)).T
    feature_max = np.tile(np.max(np.max(x_train, axis=0), axis=-1), (48, 1)).T

    x_train_n = np.array([ np.where(feature_min==feature_max, (x-feature_min) , (x-feature_min)/(feature_max-feature_min) ) for x in x_train])
    x_test_n = np.array([ np.where(feature_min==feature_max, (x-feature_min) , (x-feature_min)/(feature_max-feature_min) ) for x in x_test])

    y_train = np.array([logistic(0.5*x[0,:]*x[0,:] + 0.5*x[1,:]*x[1,:] + 0.5*x[2,:]*x[2,:]) for x in x_train_n])
    y_test = np.array([logistic(0.5 * x[0, :] * x[0, :] + 0.5 * x[1, :] * x[1, :] + 0.5 * x[2, :] * x[2, :]) for x in x_test_n])

    if not os.path.exists('./data_generator/data/simulated_data'):
        os.mkdir('./data_generator/data/simulated_data')
    save_data('./data_generator/data/simulated_data/x_train.pkl', x_train_n)
    save_data('./data_generator/data/simulated_data/y_train.pkl', y_train)
    save_data('./data_generator/data/simulated_data/x_test.pkl', x_test_n)
    save_data('./data_generator/data/simulated_data/y_test.pkl', y_test)
    save_data('./data_generator/data/simulated_data/thresholds_train.pkl', thresholds_train)
    save_data('./data_generator/data/simulated_data/thresholds_test.pkl', thresholds_test)

