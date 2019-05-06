import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
import pickle as pkl
import os

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 14 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

w, h = freqz(b, a, worN=8000)
plt.subplot(1, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


def main(n_samples, plot, Tt=48):
    signal_in = []
    thresholds = []
    trend_style=[]
    for i in range(n_samples):
        x, t,trend = generate_sample(plot, Tt=Tt)
        signal_in.append(x)
        thresholds.append(t)
        trend_style.append(trend)

    signal_in = np.array(signal_in)

    n_train = int(0.8*n_samples)
    x_train = signal_in[0:n_train,:,:]
    thresholds_train = thresholds[0:n_train]

    x_test = signal_in[n_train:,:,:]
    thresholds_test = thresholds[n_train:]

    feature_min = np.tile(np.min(np.min(x_train, axis=0), axis=-1), (Tt, 1)).T
    feature_max = np.tile(np.max(np.max(x_train, axis=0), axis=-1), (Tt, 1)).T

    x_train_n = np.array([ np.where(feature_min==feature_max, (x-feature_min) , (x-feature_min)/(feature_max-feature_min) ) for x in x_train])
    x_test_n = np.array([ np.where(feature_min==feature_max, (x-feature_min) , (x-feature_min)/(feature_max-feature_min) ) for x in x_test])

    print(np.std(np.std(x_train_n,axis=2),axis=0))
    
    x1_train_lpf = np.array([butter_lowpass_filter(x[0,:], cutoff, fs, order) for x in x_train_n])
    x2_train_lpf = np.array([butter_lowpass_filter(x[1,:], cutoff, fs, order) for x in x_train_n])
    x3_train_lpf = np.array([butter_lowpass_filter(x[2,:], cutoff, fs, order) for x in x_train_n])
    x_train_lpf = np.stack([x1_train_lpf, x2_train_lpf, x3_train_lpf],axis=1)

    x1_test_lpf = np.array([butter_lowpass_filter(x[0,:], cutoff, fs, order) for x in x_test_n])
    x2_test_lpf = np.array([butter_lowpass_filter(x[1,:], cutoff, fs, order) for x in x_test_n])
    x3_test_lpf = np.array([butter_lowpass_filter(x[2,:], cutoff, fs, order) for x in x_test_n])
    x_test_lpf = np.stack([x1_test_lpf, x2_test_lpf, x3_test_lpf],axis=1)
    
    y_train = np.array([logistic(1.5*(x[0,:]*x[0,:] +  x[1,:]*x[1,:] +  x[2,:]*x[2,:]-1)) for x in x_train_lpf])
    y_test = np.array([logistic(1.5*(x[0,:]*x[0,:] +  x[1,:]*x[1,:] +  x[2,:]*x[2,:]-1)) for x in x_test_lpf])

    if plot:
        for i in range(x_train_n.shape[0]):
            plt.plot(x_train_n[i,0,:], label='x1')
            plt.plot(x_train_n[i,1,:], label='x2')
            plt.plot(x_train_n[i,2,:], label='x3')
            plt.plot(y_train[i])
            plt.title('Sample style: %s'%(trend_style[i]))

            if isinstance(thresholds[i],np.ndarray):
                for thresh in thresholds[i]:
                    plt.axvline(x=thresh, color='grey')
            else:
                plt.axvline(x=thresholds[i], color='grey')

            plt.legend()
            plt.show()

    return x_train_n[:,:,:],y_train,x_test_n[:,:,:],y_test,thresholds_train,thresholds_test

def generate_sample(plot, Tt=48):
    seed = np.random.randint(1,100)
    # Correlation coefficients for x2=f(x1)
    coeff = [0.4,0.5,0.01]
    noise = ts.noise.GaussianNoise(std=0.05)
    trend_style = {0:'increase', 1:'decrease', 2:'hill', 3:'valley'}
    trend = np.random.randint(4)

    x1 = ts.signals.NARMA(order=15,coefficients=[.5,.5,1.5,.5],seed=seed)
    x1_ts = ts.TimeSeries(x1, noise_generator=noise)
    x1_sample, signals, errors = x1_ts.sample(np.array(range(Tt)))
    #x1_sample = x1.sample_vectorized(np.array(range(Tt)))

    
    seed = np.random.randint(1,1000)
    x2 = ts.signals.NARMA(order=15, coefficients=[.5,.5,1.5,.5], seed=seed)
    noise = ts.noise.GaussianNoise(std=0.05)
    x2_ts = ts.TimeSeries(x2,noise_generator=None)
    x2_sample,signals,errors = x2_ts.sample(np.array(range(Tt)))
    x2_sample += np.log(coeff[0]*x1_sample + coeff[1]*x1_sample*x1_sample + coeff[2]*x1_sample*x1_sample*x1_sample+1)
    #x2_sample += .5*np.random.randn(len(x2_sample))
    
    if trend==0 or trend==1:
        t = np.random.randint(20,38)
        x2_sample[t:] = x2_sample[t:] + (1 if trend==0 else -1) * np.log(1.5*np.asarray(range(Tt-t))+1.) 
    elif trend==2 or trend==3:
        t = np.sort(np.random.choice(np.arange(15,38),4))
        s = +1 if trend==2 else -1
        x2_sample[t[0]:t[1]] = x2_sample[t[0]:t[1]] + s*np.log(1.5*np.asarray(range(t[1]-t[0]))+1.)
        x2_sample[t[1]:t[2]] = x2_sample[t[1]:t[2]] + s * np.log(1.5 * np.full(x2_sample[t[1]:t[2]].shape, (t[1]-t[0])) + 1.)
        x2_sample[t[2]:t[3]] = x2_sample[t[2]:t[3]] - s*np.log(1.5*np.asarray(range(t[3]-t[2]))+1.)
    #[.5,.2,2.5,.5]
    x3 = ts.signals.NARMA(15,coefficients=[.5,.5,1.5,.5], seed=seed*5000)
    noise = ts.noise.GaussianNoise(std=0.05)
    x3_ts = ts.TimeSeries(x3, noise_generator=noise)
    x3_sample, signals, errors = x3_ts.sample(np.array(range(Tt)))
    x3_sample += 0.004*np.log(x2_sample+10)
    #x3_sample = x3.sample_vectorized(np.array(range(Tt)))

    #x1_sample -= np.mean(x1_sample) /np.std(x1_sample)
    #x2_sample -= np.mean(x2_sample) /np.std(x2_sample)
    #x3_sample -= np.mean(x3_sample) /np.std(x3_sample)
    return np.stack([x1_sample, x2_sample, x3_sample]), t, trend_style[trend]

def save_data(path,array):
    with open(path,'wb') as f:
        pkl.dump(array, f)


def logistic(x):
    return 1./(1+np.exp(-1*x))


if __name__=='__main__':
 
    n_samples = 30000
    x_train_n,y_train,x_test_n,y_test,thresholds_train,thresholds_test = main(n_samples=n_samples, plot=False)

    if not os.path.exists('./data_generator/data/simulated_data'):
        os.mkdir('./data_generator/data/simulated_data')
    save_data('./data_generator/data/simulated_data/x_train.pkl', x_train_n)
    save_data('./data_generator/data/simulated_data/y_train.pkl', y_train)
    save_data('./data_generator/data/simulated_data/x_test.pkl', x_test_n)
    save_data('./data_generator/data/simulated_data/y_test.pkl', y_test)
    save_data('./data_generator/data/simulated_data/thresholds_train.pkl', thresholds_train)
    save_data('./data_generator/data/simulated_data/thresholds_test.pkl', thresholds_test)

