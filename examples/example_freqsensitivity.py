# -*- coding: utf-8 -*-
"""
Measure the frequency sensitivity of the target microphone
==========================================================
This example will show the full path from raw audio to rms/Pa sensitivity for 
the frequency range in the playback sweep.

The steps are as followin:
    1. Remove reflections from both target and calibration mics
    2. Align target and calibration mic audio and cut out only the playback sweep
    3. Use the known spectral ??? from the calibration mic to calculate the sensitivity
    of the target microphone
    4. Test the target mic calibration using a 'validation' playback sound.

Data collected by Lena de Framond.

"""
from miccalib.deconvolution import deconvolve_linearchirp
from miccalib.deconvolution import convolve_linearchirp
from miccalib.deconvolution import eliminate_reflections_linchirp
from miccalib.utilities import cut_out_playback, make_linear_sweep, rms, dB
import miccalib.sensitivity as sensitivity

import miccalib 
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os 
import scipy.signal as signal
#%%
linearsweep_props = {'freq_start': 15e3, 'freq_stop':200, 'sweep_duration':7e-3}

# load and align the sweep from the calibration mic
calibmic_rec = os.path.join('example_data',
                            'GRAS_playback_240819_0330.wav')
fs = sf.info(calibmic_rec).samplerate
t_start, t_stop = 9.100, 9.120
b,a = signal.butter(1, 100/(fs*0.5), 'high')
calibmic_stereo, fs = sf.read(calibmic_rec,
                    start=int(fs*t_start), stop=int(fs*t_stop))
calibmic_raw = calibmic_stereo[:,0]
calibmic_audio = signal.filtfilt(b,a, calibmic_raw)

calibmic_cleanedsweep, (calib_impresp, calib_invsweep) = eliminate_reflections_linchirp(calibmic_audio, 
                                                                      fs,
                                                                      linearsweep_props,
                                                                      )
plt.figure()
plt.subplot(411)
plt.specgram(calibmic_cleanedsweep, Fs=fs, NFFT=256, noverlap=200)
plt.subplot(412)
plt.specgram(calibmic_audio, Fs=fs, NFFT=256, noverlap=200)
plt.subplot(413)
plt.plot(calibmic_cleanedsweep,)
plt.subplot(414)
plt.plot(calibmic_audio, )
#%% load and align the sweep from the target microphone 
t_start, t_stop = 6.825, 6.845 # seconds
tgt_mic_file = os.path.join('example_data','SennheiserMKE_240819_0325.wav')
tgtmic_audio_stereo, fs = sf.read(tgt_mic_file,
                           start=int(fs*t_start), stop=int(fs*t_stop))
tgtmic_audio_raw = tgtmic_audio_stereo[:,0]
tgtmic_audio = signal.filtfilt(b,a,tgtmic_audio_raw)


tgt_cleanedaudio, (tgt_impresp, tgt_invsweep) = eliminate_reflections_linchirp(tgtmic_audio, 
                                                                      fs,
                                                                      linearsweep_props,
                                                                      )

plt.figure()
plt.subplot(411)
plt.specgram(tgt_cleanedaudio, Fs=fs, NFFT=256, noverlap=200)
plt.subplot(412)
plt.specgram(tgtmic_audio, Fs=fs, NFFT=256, noverlap=200)
plt.subplot(413)
plt.plot(tgt_cleanedaudio,)
plt.subplot(414)
plt.plot(tgtmic_audio, )
#%%
linear_chirp = make_linear_sweep(linearsweep_props['sweep_duration'], fs, 
                                 (linearsweep_props['freq_start'],
                                  linearsweep_props['freq_stop']))
tgtmic_onlysweep = cut_out_playback(tgt_cleanedaudio, linear_chirp)
calibmic_onlysweep = cut_out_playback(calibmic_cleanedsweep, linear_chirp)

plt.figure()
plt.plot(tgtmic_onlysweep, label='target mic')
plt.plot(calibmic_onlysweep, label='calibration mic')
plt.legend()
#%%
# And now let's calculate the spectral RMS levels from the calibration microphone
# First compensate for the gain settings in both mics
rec_params = pd.read_excel(os.path.join('example_data',
                                        '2024-08-19_recalibration.xlsx'))

# Let's load the reference 1 kHz tone at 1 Pa from the calibrator 
onePa_rec = os.path.join('example_data', 'GRAS_1Pa_240819_0331.wav')
onePa_tone, fs = sf.read(onePa_rec)
# band-pass with a 1st order filter between 500 and 2 kHz
b,a = signal.butter(1, np.array([500,2e3])/(fs*0.5), 'bandpass')
onePa_tone_bp = signal.filtfilt(b,a,onePa_tone[:,0]) # only the 1st channel has audio

# We know from the recording paramter notes of that day that the calibration 
# microphone (a GRAS 1/4" mic) had a total gain of +56 dB. 
calibmic_total_gain = 56 # dB
calibmic_tonegain = 36
onePa_tone_gaincomp = onePa_tone_bp*10**(-calibmic_tonegain/20)

plt.figure()
plt.plot(onePa_tone_gaincomp)

#%% Calibration 
# Using the calibrator we know the GRAS mic's sensitivity at 1kHz. The cool thing
# about the gRAS mic is that it has a very flat frequency response across a wide
# range of frequncies (+/- 1-2 dB from audible to ultrasound range). We can use this
# property to now calculate the sound pressure level for the whole sound and across
# frequency bands
rms_per_Pa = rms(onePa_tone_gaincomp)

#%% Gain compensation
# Let's compensate the recordings for the gain used during recording. 
# Why compensate? This is is because we want to be able to use the same mics
# at any kind of gain value for this particular audio interface.
 
calibmic_sweep_gaincomp = calibmic_onlysweep*10**(-calibmic_total_gain/20)
tgtmic_totalgain = 36 # dB
tgtmic_sweep_gaincomp = tgtmic_onlysweep*10**(-tgtmic_totalgain/20)

# Now see the effect of gain compensation has
plt.figure()
plt.plot(tgtmic_sweep_gaincomp, label='target mic')
plt.plot(calibmic_sweep_gaincomp, label='calibration mic')
plt.title('Gain compensated audio')
plt.legend()

#%% 
# Power spectral density of the playback sweep

fftfreqs, calibfreqwise_rms = sensitivity.calc_native_freqwise_rms(calibmic_sweep_gaincomp,
                                                              fs)
plt.figure()
plt.plot(fftfreqs, calibfreqwise_rms)
plt.xlabel('Frequency, Hz', fontsize=12);plt.ylabel('Power, RMS', fontsize=12)

#%% 
# We have the RMS over frequency bands, what we need is the equivalent Pascals
# over frequency bands

calibfreqwise_Pa_rms = calibfreqwise_rms/rms_per_Pa
calibfreqwise_dBSPL = dB(calibfreqwise_Pa_rms, ref=20e-6)
valid_freqrange = fftfreqs[np.logical_and(fftfreqs>=200, fftfreqs<=15e3)]

plt.figure()
a0 = plt.subplot(211)
plt.title('Calibration mic measurements')
plt.plot(fftfreqs, calibfreqwise_Pa_rms) 
plt.ylabel('Pressure, Pa (rms)', fontsize=12)
a1 = plt.subplot(212, sharex=a0)
plt.plot(fftfreqs, calibfreqwise_dBSPL)
plt.vlines([np.min(valid_freqrange), np.max(valid_freqrange)], 0, 50, 'k')
plt.text(np.min(valid_freqrange)+100, 20, 'Sweep\nfrequency range',)
plt.ylabel('dB SPL rms, \n re 20$\mu$Pa')
plt.xlabel('Frequency, Hz')

#%%
# Notice in the dB SPL plot below there are some weird peaks especially below 200 Hz 
# The GRAS 1/4" mic used already has a noise-floor of at least 30 dB SPL. Let's
# choose a more conservative threshold of ~40 dB, and also use the information have
# that the sweep was designed from 0.2-15 kHz. Moreover, the sweep was windowed, and so 
# there is probably sufficient signal above noise level over a slightly smaller
# frequency range. 

valid_freqrange = np.logical_and(fftfreqs>=200, fftfreqs<=15e3)
above_noiselevel = calibfreqwise_dBSPL>=40
calib_clean_powerspectrum =  calibfreqwise_Pa_rms.copy()
calib_clean_powerspectrum[~np.logical_and(above_noiselevel, valid_freqrange)] =0

plt.figure()
plt.plot(fftfreqs, calib_clean_powerspectrum)
plt.title('Cleaned power spectrum - above 40 dBSPL and within sweep range')
#%%
# Since we know the sound pressure for each frequency band, we can calculate
# the sensitivity of our target microphone too.

fftfreqs, tgt_freqwiserms = sensitivity.calc_native_freqwise_rms(tgtmic_sweep_gaincomp,
                                                              fs)

tgtmic_sensitivity = tgt_freqwiserms/calib_clean_powerspectrum



#%%
plt.figure()
a0 = plt.subplot(211)
plt.plot(fftfreqs, tgtmic_sensitivity)
plt.xlabel('Frequency, Hz'); plt.ylabel('Sensitivity, a.u.rms/Pa')

a1 = plt.subplot(212)
plt.plot(fftfreqs, dB(tgtmic_sensitivity))
plt.xlabel('Frequency, Hz'); plt.ylabel('Sensitivity, dB a.u.rms/Pa')

#%%
# Now to finally 'free' the current mic sensitivity from the actual recorder used 
# we can use the data we have on the recording device to calculate the absolute
# sensitivity in Vrms/Pa. This now allows us to connect the mic to any audio interface
# to any device and begin to record. 
# We know the Clip Vrms of the recorder used here - a TASCAM portacapture X6, which has 
# a 2 dBu max Vcliprms for the channels used. 
# This allows us to go back and also calculate the Vrms/Pa sensitivity of the 
# mic - a Sennheiser MKH 416 . The manufacturer specs say the sensitivity should be at 25mV/Pa 
# , and that too very flat at +/- 1 dB across the whole frequency range of ~0.05-20 kHz. 
Vrms = 0.975*(tgtmic_sensitivity/(1/np.sqrt(2)))

plt.figure()
a00 = plt.subplot(211)
plt.plot(fftfreqs, Vrms)
plt.hlines(25e-3*10**(-2.5/20),0,fftfreqs[-1], label='+2.5 dB')
plt.hlines(25e-3*10**(2.5/20),0,fftfreqs[-1], label='-2.5 dB')
plt.hlines(25e-3,0,fftfreqs[-1], 'g', label='per manufac. specs')
plt.ylabel('Mic output, Vrms', fontsize=12)
plt.legend()

a01 = plt.subplot(212, sharex=a00)
plt.plot(fftfreqs, dB(Vrms))
plt.hlines(dB(25e-3)-2.5,0,fftfreqs[-1])
plt.hlines(dB(25e-3)+2.5,0,fftfreqs[-1])
plt.hlines(dB(25e-3),0,fftfreqs[-1])
plt.ylabel('Mic output, dB(Vrms) re 1', fontsize=12)
