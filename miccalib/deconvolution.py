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
import scipy.signal as signal 

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
        Complex conjugate of the linear sweep.
  
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
        Complex conjugate of the linear sweep

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
    #modsig2=np.real(fft.ifft(G2*np.flip(modu)))*2
    modsig2=np.real(fft.ifft(G2*np.conjugate(modu)))*2
    return modsig2


def eliminate_reflections_linchirp(audioclip, fs, linearsweep_props, **impuleresp_props):
    '''
    Eliminate reflections from a recording of a linear chirp playback
    
    Parameters
    ----------
    audioclip : (N,) np.array
    fs : int >0
        SAmplerate in Hz
    linearsweep_props : dict
        Dictionary with the following entries freq_start, freq_stop, sweep_duration.
        freq_start, freq_stop: float>=0, Hz
        sweep_duration : float >0, seconds
    impuleresp_props : dict
        A dictionary with parameters that control the peak finding and reverb-removal.
            peak_threshold : 100>float>0, optional
                The percentile threshold to define a peak after deconvolution to 
                identify where all a linear sweep is detected. Defaults to 99.
            min_interpeak_distance : float>0, optional
                The minimum time-gap between two peaks in seconds. Defaults to
                0.5e-3 seconds.
            imp_resp_halfwidth : int>0, optional
                The halfwidth of the top peak that is preserved. Defaults to 20 samples.
    
    Returns
    -------
    directpath_audioclip : (N,) np.array
        A np.array with only the direct path        
    (impulse_resp, inverted_sweep) : tuple
        A tuple with the impulse response of the audio and the inverted sweep 
        to be use to recover the original audio. 
    '''
    peak_thresh = impuleresp_props.get('peak_threshold', 99)
    imp_resp_halfwidth = impuleresp_props.get('imp_resp_halfwidth',20)
    min_interpeak_distance = impuleresp_props.get('min_interpeak_distance', 0.5e-3)
    freq_start, freq_stop = linearsweep_props['freq_start'], linearsweep_props['freq_stop']
    sweep_duration = linearsweep_props['sweep_duration']
    samples_sweep = int(fs*sweep_duration)
    sweep_rate = (freq_stop - freq_start)/sweep_duration
    
    impulse_resp, inverted_sweep = deconvolve_linearchirp(audioclip,
                                                          sweep_rate,
                                                                    fs)
    peaks_IR, _ = signal.find_peaks(impulse_resp,
                                    height=np.percentile(impulse_resp,peak_thresh),
                                    distance=int(fs*min_interpeak_distance))

    valid_samples = np.arange(peaks_IR[0]-imp_resp_halfwidth, peaks_IR[0]+imp_resp_halfwidth, )
    cleaned_impulseresp = np.zeros(impulse_resp.size)
    cleaned_impulseresp[valid_samples] = impulse_resp[valid_samples]
    directpath_audioclip = convolve_linearchirp(cleaned_impulseresp,inverted_sweep )
    
    return directpath_audioclip, (impulse_resp, inverted_sweep)



