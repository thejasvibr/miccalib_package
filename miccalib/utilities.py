# -*- coding: utf-8 -*-
"""
Utility functions to do various measurements
--------------------------------------------
Some of these functions are translations of Kristian Beedlholm's
MATLAB code.

@author: Thejasvi Beleyur & Lena de Framond
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy import fft

def DC_remove(audio):
    audio2 = audio-np.mean(audio)
    return audio2

def normalize_energy(audio):
    norm = audio/np.sqrt(np.sum(audio**2))
    return norm

def energy(audio):
    return np.sum(audio**2)

def RMS(x):
    # calculate root mean square
    rms = np.sqrt(np.mean(x**2))
    return rms

def dB(x, **kwargs):
    '''
    Parameters
    ----------
    x : float
        A value that must be brought to the decibel scale
    ref : float, optional
        A reference value to use in the decibel scale. Defaults to 1.

    Returns 
    -------
    dBx : float
        X in the decibel scale with a custom reference value if given.
    '''
    ref = kwargs.get('ref', 1)
    dBx = 20*np.log10(x/ref)
    return dBx

def undB(x, **kwargs):
    '''The opposite of the dB function.
    Brings measurements from decibels to a linear scale.
    '''
    ref = kwargs.get('ref', 1)
    dBx = ref*10**(x/20)
    return dBx

def DiffSpecs (spec_Gras_dB, f_g, spec_Sweep_dB, f_s):
    if f_s[-1]<f_g[-1]: 
        spec_Gras_dB = spec_Gras_dB[f_g<f_s[-1]]        
        f=f_s
    elif f_g[-1]<f_s[-1]: 
        spec_Sweep_dB = spec_Sweep_dB[f_s<f_g[-1]]
        f=f_g
    else:
        f = f_g
    
    f=resample_by_interpolation(f ,n_out=513)
    spec_Gras_dB = resample_by_interpolation(spec_Gras_dB ,n_out=513)
    spec_Sweep_dB = resample_by_interpolation(spec_Sweep_dB ,n_out=513)

    DiffSpec = [Sweep - Gras for Sweep, Gras in zip (spec_Sweep_dB, spec_Gras_dB)]
    return f, DiffSpec


def resample_by_interpolation(signal, input_fs=None, output_fs=None, n_out=None):
    # DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
    #             which was released under LGPL. 
    
    if n_out is None:
        scale = output_fs / input_fs
        # calculate new length of sample
        n = round(len(signal) * scale)
    elif input_fs is None:
        n=n_out

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=True),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=True),  # known positions
        signal,  # known data points
    )
    return resampled_signal


def powerspec(X, **kwargs):
    fft_X = np.fft.rfft(X)
    fft_freqs = np.fft.rfftfreq(X.size, d=1/kwargs['fs'])
    return fft_freqs, 20*np.log10(abs(fft_X))

def maxnorm_powerspec(X, **kwargs):
    fftfreqs, spectrum = powerspec(X, **kwargs)
    spectrum -= np.max(spectrum)
    return fftfreqs, spectrum

def rms(X):
    return np.sqrt(np.mean(X**2))
