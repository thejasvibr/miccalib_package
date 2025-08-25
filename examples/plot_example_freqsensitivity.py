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

Experiment design: TB, Data collected by Lena de Framond. Analysis workflow: TB, LdF
"""
from miccalib.deconvolution import deconvolve_linearchirp
from miccalib.deconvolution import convolve_linearchirp
from miccalib.deconvolution import eliminate_reflections_linchirp
from miccalib.utilities import cut_out_playback, make_linear_sweep, rms, dB
import miccalib.sensitivity as sensitivity
from miccalib.utilities import get_rms_from_fft

import miccalib 
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os 
import scipy.signal as signal
#%%
# Load audio and remove reflections
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
plt.figure(figsize=(4,7))
plt.subplot(411)
plt.title('Calibration mic sweep - raw audio')
plt.specgram(calibmic_audio, Fs=fs, NFFT=256, noverlap=200)
plt.subplot(412)
plt.plot(calibmic_audio, )
plt.xticks([])
plt.subplot(413)
plt.title('Calibration mic sweep - w/o reflections')
plt.specgram(calibmic_cleanedsweep, Fs=fs, NFFT=256, noverlap=200)
plt.subplot(414)
plt.plot(calibmic_cleanedsweep,)
plt.show()

#%%
# Load and align the sweep from the target microphone 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

plt.figure(figsize=(4,7))
plt.subplot(411)
plt.title('Target mic sweep - raw audio')
plt.plot(tgtmic_audio,)
plt.subplot(412)
plt.specgram(tgtmic_audio, Fs=fs, NFFT=256, noverlap=200)
plt.xticks([])
plt.subplot(413)
plt.title('Calibration mic sweep - w/o reflections')
plt.plot(tgt_cleanedaudio, )
plt.subplot(414)
plt.specgram(tgt_cleanedaudio, Fs=fs, NFFT=256, noverlap=200)
plt.show()
#%%
linear_chirp = make_linear_sweep(linearsweep_props['sweep_duration'], fs, 
                                 (linearsweep_props['freq_start'],
                                  linearsweep_props['freq_stop']))
tgtmic_onlysweep = cut_out_playback(tgt_cleanedaudio, linear_chirp)
calibmic_onlysweep = cut_out_playback(calibmic_cleanedsweep, linear_chirp)

plt.figure(figsize=(4,4))
plt.title('Linear sweeps w/o reflections')
plt.plot(tgtmic_onlysweep, label='target mic')
plt.plot(calibmic_onlysweep, label='calibration mic')
plt.legend()
plt.show()

#%% 
# Calibration tone: connecting a.u. RMS to Pascals 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# How do we connect the calibration mic's a.u. rms to 'real-world' Pascals?
# We will do it using a calibrator that produces a 1 kHz tone at 1 Pa sound pressure. 
# The cool thing about the the GRAS mic used here is that it has a very flat frequency response across a wide
# range of frequncies (+/- 1-2 dB from audible to ultrasound range). The a.u. rms/Pa sensitivity 
# at 1 kHz can be fairly approximated to be the same up to 100 kHz!

#%%
# A note about 'a.u. rms'
# ^^^^^^^^^^^^^^^^^^^^^^^
# a.u. is short for arbitrary units. If there is 'arbitrary' in the RMS
# it means the rms values are specific to that recorder!

# Let's load the reference 1 kHz tone at 1 Pa from the calibrator 
onePa_rec = os.path.join('example_data', 'GRAS_1Pa_240819_0331.wav')
onePa_tone, fs = sf.read(onePa_rec)
# band-pass with a 1st order filter between 500 and 2 kHz
b,a = signal.butter(1, np.array([200])/(fs*0.5), 'highpass')
onePa_tone_bp = signal.filtfilt(b,a,onePa_tone[:,0]) # only the 1st channel has audio
calibmic_tonegain = 36
onePa_tone_gaincomp = onePa_tone_bp*10**(-calibmic_tonegain/20)

plt.figure(figsize=(5,3))
plt.title('Calib. mic: 1Pa tone @1 kHz (5 ms snippet)')
plt.plot(onePa_tone_gaincomp[:int(fs*0.005)])
plt.ylabel('Arbitrary units', fontsize=12);
plt.show()

rms_per_Pa = rms(onePa_tone_gaincomp)
print(f'The calibration mic has a sensitivity of: {np.around(rms_per_Pa,4)} a.u.rms/Pa')

#%%
# Gain compensation
# ~~~~~~~~~~~~~~~~~
# Let's compensate the recordings for the gain used during recording. 
# Why compensate? This is is because we want to be able to use the same mics
# at any kind of gain value for this particular audio interface.
# We know from the recording paramter notes of that day that the calibration 
# microphone (a GRAS 1/4" mic) had a total gain of +56 dB. 
calibmic_total_gain = 56 # dB

calibmic_sweep_gaincomp = calibmic_onlysweep*10**(-calibmic_total_gain/20)
tgtmic_totalgain = 36 # dB
tgtmic_sweep_gaincomp = tgtmic_onlysweep*10**(-tgtmic_totalgain/20)

# Now see the effect of gain compensation has
plt.figure(figsize=(7,3))
plt.plot(tgtmic_sweep_gaincomp, label='target mic')
plt.plot(calibmic_sweep_gaincomp, label='calibration mic')
plt.title('Gain compensated audio');plt.ylabel('Arbitrary units', fontsize=12);
plt.legend()

#%% 
# Power spectral density of the playback sweep - a.u. rms over frequency
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To calculate how much energy is there in each frequency band, we will 
# get the RMS/frequency band through an FFT using Parseval's theorem. 

fftfreqs, calibfreqwise_rms = sensitivity.calc_native_freqwise_rms(calibmic_sweep_gaincomp,
                                                              fs)
plt.figure()
plt.plot(fftfreqs, calibfreqwise_rms)
plt.xlabel('Frequency, Hz', fontsize=12);plt.ylabel('Power, a.u. RMS', fontsize=12)
plt.title('Calib. mic: power spectrum')
#%% 
# We have the RMS over frequency bands, what we need is the equivalent Pascals
# over frequency bands: :math:`Pascals_{rms} \ across \ freqs. = \frac{signal\ RMS \ across \ freqs.}{sensitivity \ across \ freqs.}`

calibfreqwise_Pa_rms = calibfreqwise_rms/rms_per_Pa
calibfreqwise_dBSPL = dB(calibfreqwise_Pa_rms, ref=20e-6)
valid_freqrange = fftfreqs[np.logical_and(fftfreqs>=200, fftfreqs<=15e3)]

plt.figure()
a0 = plt.subplot(211)
plt.title('Calib.mic: sweep power spectrum')
plt.plot(fftfreqs, calibfreqwise_Pa_rms) 
plt.ylabel('Pressure, Pa (rms)', fontsize=12)
a1 = plt.subplot(212, sharex=a0)
plt.plot(fftfreqs, calibfreqwise_dBSPL)
plt.vlines([np.min(valid_freqrange), np.max(valid_freqrange)], 0, 50, 'k')
plt.text(np.min(valid_freqrange)+100, 20, 'Sweep\nfrequency range',)
plt.ylabel('dB SPL rms, \n re 20$\mu$Pa')
plt.xlabel('Frequency, Hz')

#%%
# Cleaning up the power-spectrum: accounting for signal windowing and calib. mic noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

plt.figure(figsize=(7,5))
plt.plot(fftfreqs, calib_clean_powerspectrum)
plt.xlim(1e3,16e3);plt.ylabel('Sound pressure, Pa (rms)', fontsize=12)
plt.xlabel('Frequency, Hz', fontsize=12)
plt.title('Cleaned power spectrum - above 40 dBSPL and within sweep range')
#%%
# Since we know the sound pressure for each frequency band, we can calculate
# the sensitivity of our target microphone too.
fftfreqs, tgt_freqwiserms = sensitivity.calc_native_freqwise_rms(tgtmic_sweep_gaincomp,
                                                              fs)
tgtmic_sensitivity = tgt_freqwiserms/calib_clean_powerspectrum


#%%
tgtmic_sensitivity[np.isinf(tgtmic_sensitivity)] = np.nan

plt.figure(figsize=(7,9))
a0 = plt.subplot(211)
plt.text(3000, np.nanmax(tgtmic_sensitivity)-0.001,'Target mic: sensitivity', 
         fontsize=12)
plt.plot(fftfreqs, tgtmic_sensitivity)
plt.ylabel('Sensitivity, a.u.rms/Pa')
plt.xlim(1e3,16e3)
a1 = plt.subplot(212, sharex=a0)
plt.plot(fftfreqs, dB(tgtmic_sensitivity))
plt.xlabel('Frequency, Hz'); plt.ylabel('Sensitivity, dB a.u.rms/Pa')
plt.xlim(1e3,16e3)
plt.tight_layout()
#%% 
# What is the absolute pressure in Pa or dB SPL?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Knowing the sensitivity, let's calculate the received levels. 
# sensitivity -> au rms/ Pa 
# we want Parms = aurms x Pa/aurms

# let's load up another sound snippet, and this time just quantify the received levels 'raw'
calibmic_stereo_test, fs = sf.read(calibmic_rec,
                    start=int(fs*17.276), stop=int(fs*17.296))
calibmic_raw_test = calibmic_stereo_test[:,0]
calib_test_sweep = cut_out_playback(calibmic_raw_test, linear_chirp)
calib_test_sweep *= 10**(-calibmic_total_gain/20) # compensate for gain


tgtmic_audio_stereo, fs = sf.read(tgt_mic_file,
                           start=int(fs*23.174), stop=int(fs*23.196))
tgtmic_audio_raw = tgtmic_audio_stereo[:,0]
tgt_test_sweep = cut_out_playback(tgtmic_audio_raw, linear_chirp)
tgt_test_sweep *= 10**(-tgtmic_totalgain/20) # compensate for gain

plt.figure()
aa = plt.subplot(311)
plt.plot(calib_test_sweep, label='calib. mic');plt.plot(tgt_test_sweep, label='tgt mic')
plt.legend()
ab = plt.subplot(312)
plt.specgram(calib_test_sweep, Fs=fs, NFFT=128, noverlap=120)
ac = plt.subplot(313)
plt.specgram(tgt_test_sweep, Fs=fs, NFFT=128, noverlap=120)

# And now let's calculate the dB SPL from the calib mic
fftfreqs, calibmic_sweep_freqrms = sensitivity.calc_native_freqwise_rms(calib_test_sweep,
                                                              fs)
calibmic_sweep_rmseq = get_rms_from_fft(fftfreqs, calibmic_sweep_freqrms, freq_range=[2e3,14e3] )
calibmic_sweep_Parmseq = calibmic_sweep_rmseq/rms_per_Pa
print(f'Calib {dB(calibmic_sweep_Parmseq/20e-6)} dB SPL re 20muPa rms')
# And now using the sensitivity of the target mic

tgt_fftfreqs, tgt_test_freqrms = sensitivity.calc_native_freqwise_rms(tgt_test_sweep,
                                                              fs)
tgt_test_freqParms = tgt_test_freqrms/tgtmic_sensitivity
tgt_test_freqParms[np.isnan(tgt_test_freqParms)] = 0
tgt_test_overallParms = get_rms_from_fft(tgt_fftfreqs, tgt_test_freqParms, freq_range=[2e3,14e3] )
print(f'Target {dB(tgt_test_overallParms/20e-6)} dB SPL re 20muPa rms')

#%%
# Calculating 'absolute' sensitivity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

plt.figure(figsize=(7,9))
a00 = plt.subplot(211)
plt.title('Target mic: absolute sensitivity (Sennheiser MKH 416)')
plt.plot(fftfreqs, Vrms)
plt.hlines(25e-3*10**(-2.5/20),0,fftfreqs[-1], 'k', label='+2.5 dB')
plt.hlines(25e-3*10**(2.5/20),0,fftfreqs[-1], 'k',label='-2.5 dB')
plt.hlines(25e-3,0,fftfreqs[-1], 'g', label='per manufac. specs')
plt.ylabel('Mic output, Vrms', fontsize=12)
plt.legend()

a01 = plt.subplot(212, sharex=a00)
plt.plot(fftfreqs, dB(Vrms))
plt.hlines(dB(25e-3)-2.5,0,fftfreqs[-1])
plt.hlines(dB(25e-3)+2.5,0,fftfreqs[-1])
plt.hlines(dB(25e-3),0,fftfreqs[-1])
plt.ylabel('Mic output, dB(Vrms) re 1', fontsize=12)
plt.xlabel('Frequency, Hz')
