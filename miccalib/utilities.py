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
import scipy.signal as signal


def make_linear_sweep(duration, fs, startend_freqs):
    """Simple wrapper to make a linear sweep

    Parameters
    ----------
    duration : float>0
        Duration in seconds
    fs : float>0
        Sample rate in Hz.
    startend_freqs : tuple with floats
        Start and end frequency in Hz in this order: (start, end)

    Returns
    -------
    linear_sweep : np.array
    """
    t = np.linspace(0, duration, int(fs * duration))
    fstart, fend = startend_freqs
    linear_sweep = signal.chirp(t, fstart, t[-1], fend)
    return linear_sweep


def DC_remove(audio):
    audio2 = audio - np.mean(audio)
    return audio2


def normalize_energy(audio):
    norm = audio / np.sqrt(np.sum(audio**2))
    return norm


def energy(audio):
    return np.sum(audio**2)


def RMS(x):
    # calculate root mean square
    rms = np.sqrt(np.mean(x**2))
    return rms


def dB(x, **kwargs):
    """
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
    """
    ref = kwargs.get("ref", 1)
    dBx = 20 * np.log10(x / ref)
    return dBx


def undB(x, **kwargs):
    """The opposite of the dB function.
    Brings measurements from decibels to a linear scale.
    """
    ref = kwargs.get("ref", 1)
    dBx = ref * 10 ** (x / 20)
    return dBx


def cut_out_playback(audiorec, playback_signal):
    """
    Cross-correlates the audio recording with the playback signal
    and cuts out the relevant section.

    Parameters
    ----------
    audiorec, playback_signal : np.array
        audiorec is the microphone audio. playback_signal is the synthetic
        signal played back.

    Returns
    -------
    cutout_audiorec : np.array

    """
    if playback_signal.size >= audiorec.size:
        raise ValueError(
            f"Playback signal is {playback_signal.size} samples\
                         long, and longer than the audio recording of {audiorec.size}"
        )
    cc = signal.correlate(audiorec, playback_signal)
    peak = np.argmax(cc)
    left_end = peak - playback_signal.size
    right_end = left_end + playback_signal.size
    cutout_audiorec = audiorec[left_end:right_end]
    return cutout_audiorec


def DiffSpecs(spec_Gras_dB, f_g, spec_Sweep_dB, f_s):
    if f_s[-1] < f_g[-1]:
        spec_Gras_dB = spec_Gras_dB[f_g < f_s[-1]]
        f = f_s
    elif f_g[-1] < f_s[-1]:
        spec_Sweep_dB = spec_Sweep_dB[f_s < f_g[-1]]
        f = f_g
    else:
        f = f_g

    f = resample_by_interpolation(f, n_out=513)
    spec_Gras_dB = resample_by_interpolation(spec_Gras_dB, n_out=513)
    spec_Sweep_dB = resample_by_interpolation(spec_Sweep_dB, n_out=513)

    DiffSpec = [Sweep - Gras for Sweep, Gras in zip(spec_Sweep_dB, spec_Gras_dB)]
    return f, DiffSpec


def resample_by_interpolation(signal, input_fs=None, output_fs=None, n_out=None):
    # DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py,
    #             which was released under LGPL.

    if n_out is None:
        scale = output_fs / input_fs
        # calculate new length of sample
        n = round(len(signal) * scale)
    elif input_fs is None:
        n = n_out

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
    fs = kwargs.get("fs", None)
    fft_X = np.fft.rfft(X)
    fft_freqs = np.fft.rfftfreq(X.size, d=1 / fs)
    return fft_freqs, 20 * np.log10(abs(fft_X))


def maxnorm_powerspec(X, **kwargs):
    fftfreqs, spectrum = powerspec(X, **kwargs)
    spectrum -= np.max(spectrum)
    return fftfreqs, spectrum

# bin width clarification https://stackoverflow.com/questions/10754549/fft-bin-width-clarification
def get_rms_from_fft(freqs, spectrum, **kwargs):
    '''Use Parseval's theorem to get the RMS level of each frequency component
    This only works for RFFT spectrums!!!
    
    Parameters
    ----------
    freqs : (Nfreqs,) np.array >0 values
    spectrum : (Nfreqs,) np.array (complex)
    freq_range : (2,) array-like
        Min and max values
    
    Returns 
    -------
    root_mean_squared : float
        The RMS of the signal within the min-max frequency range
   
    '''
    minfreq, maxfreq = kwargs['freq_range']
    relevant_freqs = np.logical_and(freqs>=minfreq, freqs<=maxfreq)
    spectrum_copy = spectrum.copy()
    spectrum_copy[~relevant_freqs] = 0
    if np.any(np.iscomplex(spectrum_copy)):
        raise ValueError('Complex spectral power detected -  invalid input. ')
    root_mean_squared = np.sqrt(np.sum(spectrum_copy**2))
    return root_mean_squared


def rms(X):
    return np.sqrt(np.mean(X**2))
