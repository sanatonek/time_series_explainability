import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit



def generate_sample(i):
    coeff = [0.4,0.5,0.01]
    noise = ts.noise.GaussianNoise(std=0.3)

    x1 = ts.signals.NARMA(seed=i)
    x1_ts = ts.TimeSeries(x1, noise_generator=noise)
    x1_sample, signals, errors = x1_ts.sample(np.array(range(48)))
    #x1_sample = x1.sample_vectorized(np.array(range(48)))

    t = np.random.randint(20,38)
    print(t)
    x2_sample = coeff[0]*x1_sample + coeff[1]*x1_sample*x1_sample + coeff[2]*x1_sample*x1_sample*x1_sample
    x2_sample[t:] += np.log(0.3*np.asarray(range(48-t))+1.)
    x2_sample += .5*np.random.randn(len(x2_sample))

    x3 = ts.signals.NARMA(10,[.5,.2,2.5,.5], seed=i*5000)
    x3_ts = ts.TimeSeries(x3, noise_generator=noise)
    x3_sample, signals, errors = x3_ts.sample(np.array(range(48)))
    #x3_sample = x3.sample_vectorized(np.array(range(48)))

    y = expit(.5*x1_sample*x1_sample + 0.5*x2_sample*x2_sample + 0.5*x3_sample*x3_sample)

    plt.plot(x1_sample, label='x1')
    plt.plot(x2_sample, label='x2')
    plt.plot(x3_sample, label='x3')
    plt.plot(y)
    plt.legend()
    plt.show()

for i in range(10):
    generate_sample(i)