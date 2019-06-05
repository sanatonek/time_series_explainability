import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
import pickle as pkl
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.colors as mcolors

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
cutoff = 3.6 # desired cutoff frequency of the filter, Hz

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


def main(n_samples, plot, Tt=200):
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

    if 0:
        scaler = StandardScaler()
        x_train_flat = scaler.fit_transform(np.reshape(x_train,[x_train.shape[0],-1]))
        x_train_n = np.reshape(x_train_flat,x_train.shape)
        x_test_flat = scaler.transform(np.reshape(x_test,[x_test.shape[0],-1]))
        x_test_n = np.reshape(x_test_flat,x_test.shape)
    else:
        x_train_n = x_train
        x_test_n = x_test

    #print(np.std(np.std(x_train_n,axis=2),axis=0))
    
    x1_train_lpf = np.array([butter_lowpass_filter(x[0,:], cutoff, fs, order) for x in x_train_n])
    x2_train_lpf = np.array([butter_lowpass_filter(x[1,:], cutoff, fs, order) for x in x_train_n])
    x3_train_lpf = np.array([butter_lowpass_filter(x[2,:], cutoff, fs, order) for x in x_train_n])
    x_train_lpf = np.stack([x1_train_lpf, x2_train_lpf, x3_train_lpf],axis=1)

    x1_test_lpf = np.array([butter_lowpass_filter(x[0,:], cutoff, fs, order) for x in x_test_n])
    x2_test_lpf = np.array([butter_lowpass_filter(x[1,:], cutoff, fs, order) for x in x_test_n])
    x3_test_lpf = np.array([butter_lowpass_filter(x[2,:], cutoff, fs, order) for x in x_test_n])
    x_test_lpf = np.stack([x1_test_lpf, x2_test_lpf, x3_test_lpf],axis=1)
    
    n_phases = 3
    
    enc = OneHotEncoder(categories='auto')
    enc.fit(np.reshape(np.array([0,1,2]),[-1,1]))
    
    y_train_backup=[]
    y_train_sample=[]
    ground_truth_importance_train=[]
    for n,x in enumerate(x_train_lpf):
        
        #based on past and max
        y_tar_backup=[]
        y_tar_sample=[]
        for t in range(x.shape[1]):
            y_tar_backup.append(logistic(x[0,t]))
            y_tar_sample.append(logistic(x[1,t]))
        y_train_backup.append(y_tar_backup)
        y_train_sample.append(y_tar_sample)
    
    y_train_backup = np.array(y_train_backup)
    y_train_sample = np.array(y_train_sample)
             
    if plot:
        
        k=0
        for i in range(x_train_n.shape[0]):
            #plt.figure()
            #plt.plot()
            f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(12, 8)) 

            ax1.plot(x_train_n[i,0,:], label='generated',color=mcolors.TABLEAU_COLORS['tab:blue'],linewidth=3.0, marker='<',markerfacecolor='m')
            #ax1.plot(x_train_n[i,0,:], label='generated',color=mcolors.CSS4_COLORS['tab:blue'],linewidth=3.0)
            ax1.set_ylim([-3.5,3.5])
            ax1.grid(True,linestyle=':')
            for tic in ax1.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in ax1.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
                
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["bottom"].set_visible(False)
            ax1.spines["left"].set_visible(False)

            ax2.plot(x_train_n[i,1,:], label='observed',color=mcolors.TABLEAU_COLORS['tab:orange'],linewidth=3.0,marker='<',markerfacecolor='m')
            #ax2.plot(x_train_n[i,1,:], label='observed',color=mcolors.CSS4_COLORS['tab:orange'],linewidth=3.0)
            ax2.set_ylim([-3.5,3.5])
            ax2.grid(True,linestyle=':')
            for tic in ax2.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in ax2.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            

            ax3.plot(y_train_backup[i,:], label='generated y',color=mcolors.TABLEAU_COLORS['tab:green'], ls='-',linewidth=3.5)
            
            ax3.set_ylim([0,1])
            ax4 = ax3.twinx()
            ax4.plot(y_train_sample[i,:], label='observed y',color=mcolors.TABLEAU_COLORS['tab:brown'], ls='-',linewidth=3.5)
            #ax3.grid(False,linestyle=':')
            for tic in ax3.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in ax3.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False

            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)
            ax3.spines["left"].set_visible(False)
            
             
            #ax4.set_ylim([0,1])
            #ax4.grid(False,linestyle=':')
            for tic in ax4.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in ax4.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)
            ax4.spines["bottom"].set_visible(False)
            ax4.spines["left"].set_visible(False)
            plt.show()
            
            extent = ax1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            f.savefig('ax1_figure_expanded_%d.pdf'%(i), bbox_inches=extent.expanded(1.1, 1.2))
            extent = ax2.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            f.savefig('ax2_figure_expanded_%d.pdf'%(i), bbox_inches=extent.expanded(1.1, 1.2))
            extent = ax3.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            f.savefig('ax3_figure_expanded_%d.pdf'%(i), bbox_inches=extent.expanded(1.1, 1.2))
            extent = ax4.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            f.savefig('ax4_figure_expanded_%d.pdf'%(i), bbox_inches=extent.expanded(1.1, 1.2))
            
    #return x_train_n[:,:2,:],y_train,x_test_n[:,:2,:],y_test,thresholds_train,thresholds_test, ground_truth_importance_train[:,:], ground_truth_importance_test[:,:]

def generate_sample(plot, Tt=48):
    seed = np.random.randint(1,100)
    # Correlation coefficients for x2=f(x1)
    coeff = [0.4,0.5,0.01]
    noise = ts.noise.GaussianNoise(std=0.01)
    #trend_style = {0:'increase', 1:'decrease', 2:'hill', 3:'valley'}
    trend_style = {2:'hill', 3:'valley'}
    trend = np.random.randint(4)
    trend = 2
    
    seed = np.random.randint(1,1000)
    x2 = ts.signals.GaussianProcess(variance=0.2)
    noise = ts.noise.GaussianNoise(std=0.01)
    x2_ts = ts.TimeSeries(x2,noise_generator=None)
    x2_sample,signals,errors = x2_ts.sample(np.array(range(Tt)))

    x2_backup = x2_sample.copy()
    if trend==0 or trend==1:
        t = np.random.randint(20,38)
        x2_sample[t:] = x2_sample[t:] + (1 if trend==0 else -1) * np.log(1.5*np.asarray(range(Tt-t))+1.) 
    elif trend==2 or trend==3:
        t=[]
        t_st = np.random.choice(np.arange(130,140),1)[0]
        t.append(t_st)
        t.append(t_st+2)
        t_end = np.random.choice(np.arange(t_st+5,t_st+5+8),1)[0]
        t_env = t_end-2
        t.append(t_env)
        t.append(t_end)
        t = np.array(t)
        
        s = +1 if trend==2 else -1
        x2_sample[t[0]:t[3]] = x2_sample[t[0]:t[3]] + s * np.log(3.2* np.full(x2_sample[t[0]:t[3]].shape, (t[1]-t[0])) + 1.)

    noise = ts.noise.GaussianNoise(std=0.01)
    x3 = ts.signals.NARMA(order=5,coefficients=[.25,.25,0.05,.25],seed=seed)
    x3_ts = ts.TimeSeries(x3, noise_generator=noise)
    x3_sample, signals, errors = x3_ts.sample(np.array(range(Tt)))
    
    return np.stack([x2_backup, x2_sample, x3_sample]), t, trend_style[trend]

def save_data(path,array):
    with open(path,'wb') as f:
        pkl.dump(array, f)


def logistic(x):
    return 1./(1+np.exp(-1*x))


if __name__=='__main__':
 
    n_samples = 30000
    main(n_samples=n_samples, plot=False)

