# -*- coding: utf-8 -*-
"""
Why remove reflections?
=======================
This example will compare the effect of using 'raw' audio as recorded vs 
'cleaned' audio with reflections removed. This is actually inspired by real-life
when Lena & I first began figuring out mic calibrations using the substitution method. 

"""
import miccalib 
from miccalib.deconvolution import deconvolve_linearchirp
from miccalib.deconvolution import convolve_linearchirp
from miccalib.utilities import cut_out_playback, make_linear_sweep, rms, dB
from miccalib.deconvolution import eliminate_reflections_linchirp
import miccalib.sensitivity as sensitivity
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np 
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

linear_chirp = make_linear_sweep(linearsweep_props['sweep_duration'], fs, 
                                 (linearsweep_props['freq_start'],
                                  linearsweep_props['freq_stop']))
windowed_linearchirp = linear_chirp*signal.windows.tukey(linear_chirp.size)

#%%
# First let's cut out the sweep only using the raw audio with reflections
calibmic_only_sweep = cut_out_playback(calibmic_audio, windowed_linearchirp)
fftfreq, freqrms = sensitivity.calc_native_freqwise_rms(calibmic_only_sweep, fs)

#%% 
# Now let's remove the reflections and cut out the sweep. 

calibmic_cleanedsweep, (calib_impresp, calib_invsweep) = eliminate_reflections_linchirp(calibmic_audio, 
                                                                      fs,
                                                                      linearsweep_props,
                                                                      )
calibmiccleaned_only_sweep = cut_out_playback(calibmic_cleanedsweep, windowed_linearchirp)
fftfreq_clean, freqrms_clean = sensitivity.calc_native_freqwise_rms(calibmiccleaned_only_sweep, fs)
#%%
# Comparing the effects of removing reflections and not
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now let's look at the effect doing the reflection removal and not doing it. 
# Notice how 'spiky' the power spectrum of the 'raw' audio sweep is. This is because of all 
# the reflections adding up with the direct path at various delays and messing up the 
# expected smooth spectrum. Also notice some energy past 15 kHz - which shouldn't even be there
# this is all from the 2nd harmonic.
# 
# In contrast, look at the recorded audio with reflections (and harmonics) removed. The 
# obtained output has the smooth profile expected from the synthetic linear chirp that was
# actually played back. 

fig = plt.figure(figsize=(7,7))
gs = plt.GridSpec(2, 2, figure=fig)
a00 = fig.add_subplot(gs[0,0])
plt.title('Raw audio: sweep + reflections')
plt.plot(windowed_linearchirp*np.max(calibmic_only_sweep), label='playback signal\n(scaled)')
plt.plot(calibmic_only_sweep, label='recorded signal')
plt.legend()
plt.ylabel('Amplitude, a.u.', fontsize=12)

a01 = fig.add_subplot(gs[1,0])
plt.plot(fftfreq, freqrms)
plt.ylabel('Power, a.u. RMS', fontsize=12);plt.xlabel('Frequency, Hz', fontsize=12)

a10 = fig.add_subplot(gs[0,1])
plt.title('Cleaned audio: sweep w/o reflections')
plt.plot(windowed_linearchirp*np.max(calibmic_only_sweep), label='playback signal')
plt.plot(calibmiccleaned_only_sweep, label='recorded signal')


a11 = fig.add_subplot(gs[1,1], sharey=a01)
plt.plot(fftfreq, freqrms_clean)
plt.xlabel('Frequency, Hz', fontsize=12)
plt.tight_layout()
plt.show()
#%%
# Summary 
# ~~~~~~~
# This is a short illustration of why reflection-removal is important. Now, imagine we hadn't
# removed the reflections and just gone ahead and done the target mic sensitivity calculations. 
# The sensitivity curve would also look equally jagged considering both target and calibration
# mic audio would have jagged spectra - instead of the expected smoothly varying spectra. 
# 
# There is another way to verify the importance of removing reflections - which is to use a
# a very short chirp (~3 ms) - and also include a) a little of audio before the chirp vs b) a little bit of audio 
# after the chirp. Try this out and you will see the difference the reflections make. 



