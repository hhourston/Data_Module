import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy import cwt, wavs
from ssqueezepy.visuals import plot
from ssqueezepy.experimental import scale_to_freq


def plot_cwt(signal, wavelet, t, nv, N, title):
    """Plot continuous wavelet transform"""
    
    # Time step
    dt = t[1] - t[0] if t is not None else 1
    fs = 1/dt if dt is not None else None
    
    Wx, scales = cwt(
        signal,
        wavelet=wavelet,
        fs=fs,
        nv=nv
    )
    
    freqs = scale_to_freq(scales, wavelet=wavelet, fs=fs, N=N)

    plt.figure(figsize=(4, 3), layout='constrained')
    plt.pcolormesh(t, freqs, np.abs(Wx), shading='auto')
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.colorbar(label='|CWT|')
    plt.show()
    return Wx, scales
