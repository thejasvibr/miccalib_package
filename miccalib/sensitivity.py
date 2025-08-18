# -*- coding: utf-8 -*-
"""
Microphone sensitivity and related
----------------------------------
Power spectra, spectral RMS and sensitivity calculations


@author: Thejasvi Beleyur, 2025
"""
import numpy as np 
from scipy.interpolate import interp1d

def calc_native_freqwise_rms(X, fs):
    '''
    Converts the FFT spectrum into a band-wise rms output. 
    The frequency-resolution of the spectrum/audio size decides
    the frequency resolution in general - so a longer sound is
    often better. 
    
    Parameters
    ----------
    X : np.array
        Audio
    fs : int
        Sampling rate in Hz
    
    Returns 
    -------
    fftfreqs, freqwise_rms : np.array
        fftfreqs holds the frequency bins from the RFFT
        freqwise_rms is the RMS value of each frequency bin. 
    '''
    rfft = np.fft.rfft(X)
    fftfreqs = np.fft.rfftfreq(X.size, 1/fs)
    # now calculate the rms per frequency-band
    freqwise_rms = []
    for each in rfft:
        mean_sq_freq = np.sum(abs(each)**2)/rfft.size
        rms_freq = np.sqrt(mean_sq_freq/(2*rfft.size-1))
        freqwise_rms.append(rms_freq)
    freqwise_rms = np.array(freqwise_rms)
    return fftfreqs, freqwise_rms


# Make an interpolation function 
def interpolate_freq_response(mic_freq_response, new_freqs):
    ''' 
    
    
    Parameters
    ----------
    mic_freq_response : tuple/list
        A tuple/list with two entries: (centrefreqs, centrefreq_RMS).
        
    new_freqs : list/array-like
        A set of new centre frequencies that need to be interpolated to. 

    Returns 
    -------
    tgtmicsens_interp : 
        
    Attention
    ---------
    Any frequencies outside of the calibration range will automatically be 
    assigned to the lowest sensitivity values measured in the input centrefreqs
    
    '''
    centrefreqs, mic_sensitivity = mic_freq_response 
    tgtmic_sens_interpfn = interp1d(centrefreqs, mic_sensitivity,
                                    kind='cubic', bounds_error=False,
                                    fill_value=np.min(mic_sensitivity))
    # interpolate the sensitivity of the mic to intermediate freqs
    tgtmicsens_interp = tgtmic_sens_interpfn(new_freqs)
    return tgtmicsens_interp

