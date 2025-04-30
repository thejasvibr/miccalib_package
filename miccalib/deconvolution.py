# -*- coding: utf-8 -*-
"""
De-&-convolution functions 
-----------------------
Deconvolution and convolution functions to generate cleaner, reflection-free
estimates of known playback signals. The general steps are to:
    
    1. Identify where the signals are detected in time as an 'impulse response' (direct path & reflections)
    2. Clean the 'impulse response' manually to suppress all reflections, and keep only the direct path
    3. Convolve the cleaned impulse response with the known signal to get the reflection-free, deconvolved audio


Code origins
~~~~~~~~~~~~
deconvolve_linearchirp & convolve_linearchirp were adapted by TB 
based on Lena de Framond's translation of Kristian Beedlholm's MATLAB code. 

@author: Thejasvi Beleyur, Lena de Framond, 2025
"""
import numpy as np 
from scipy import fft

def deconvolve_linearchirp(audio, sweeprate, fs):
    '''
    Identify where in time the linear chirp occurs in the 
    input audio file 
    
    To obtain a 'clean' audio later, peak-selection must be separately done!
    
    Parameters
    ----------
    audio : np.array 
    sweeprate : float>0
        A sweeprate in kHz/s. A positive sweeprate means an 'ascending'
        chirp, and -ve sweeprate means a 'descending' chirp
    fs : float>0
        Sampling rate in Hz. 

    Returns
    -------
    modsig: np.array
        The impulse-response of the audio recording showing where all
        the playback signal was detected in time.
    
    modu : np.array 
        The inverted version of linear-chirp for convolution later.
  
    See Also
    --------
    miccalib.deconvolution.convolve_linearchirp
  
    '''
    L=len(audio)
    G=fft.fft(audio)
    f = np.arange(0, L, 1)/L*fs # frequencies of the FFT from 0-fs
    t = f*L/fs**2
    gdl=f/sweeprate

    modu=np.exp(1j*np.cumsum(gdl)*2*np.pi/L*fs)
    modu[int(len(modu)/2):-1]=0
    modu[-1]=0

    modsig=np.real(fft.ifft(G*modu))*2

    return modsig, modu

def convolve_linearchirp(modsig, modu):
    '''
    To be used in conjunction with deconvolve_linearchirp
    
    Parameters
    ----------
    modsig : np.array
        The 'cleaned' impulse-response of the recording. 
    modu : np.array
        The inverted linear-chirp to convolve the clean impulse-response with

    Returns 
    -------
    modsig2 :  np.array
        The convolved audio with the playback signal without any 
    reflections.

    See Also
    --------   
    miccalib.deconvolution.deconvolve_linearchirp    
    '''
    G2 = fft.fft(modsig)
    modsig2=np.real(fft.ifft(G2*np.flip(modu)))*2
    return modsig2
