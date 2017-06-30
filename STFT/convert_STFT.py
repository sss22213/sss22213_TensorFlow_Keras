import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd 

input_data=pd.read_csv('Train.csv')
x_matrix=input_data.as_matrix()

fs=44100
divid=8
noverlap=2
for x in range(1000):
    window=signal.get_window('hamming', 8)
    f, t, Sxx = signal.spectrogram(x_matrix[x], fs,window,divid,noverlap)
    fig,ax = plt.subplots(1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.pcolormesh(t, f, Sxx)
    plt.savefig(str(x)+'.png')
    plt.close()