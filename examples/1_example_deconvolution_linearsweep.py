# -*- coding: utf-8 -*-
"""
Example runthrough deconvolution
--------------------------------


@author: Thejasvi
"""
import miccalib 
from miccalib.deconvolution import deconvolve_linearchirp
from miccalib.deconvolution import convolve_linearchirp
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np 
import os 
import scipy.signal as signal

calibmic_rec = os.path.join('example_data',
                            'GRAS_playback_240819_0330.wav')
fs = sf.info(calibmic_rec).samplerate
t_start, t_stop = 9.105, 9.120 # 7 ms sweep
#t_start, t_stop = 8.895, 8.915 # 5 ms sweep
#t_start, t_stop = 8.695, 8.710 # 3 ms sweep


b,a = signal.butter(1, 100/(fs*0.5), 'high')


audio_stereo, fs = sf.read(calibmic_rec,
                    start=int(fs*t_start), stop=int(fs*t_stop))
audio_raw = audio_stereo[:,0]

audio = signal.filtfilt(b,a, audio_raw)


freq_start, freq_stop = 15e3, 200
sweep_duration = 7e-3
samples_sweep = int(fs*sweep_duration)
sweep_rate = (freq_stop - freq_start)/sweep_duration

impulse_resp, inverted_sweep = deconvolve_linearchirp(audio, sweep_rate, fs)

# do some

plt.figure()
plt.subplot(311)
plt.plot(impulse_resp)
plt.subplot(312)
plt.specgram(audio, Fs=fs, NFFT=256, noverlap=255)
plt.subplot(313)
plt.plot(audio)


#%% Find the first big peak, and suppress all other peaks
peaks_IR, _ = signal.find_peaks(impulse_resp,
                                height=np.percentile(impulse_resp, 99),
                                distance=int(fs*0.5e-3))

directpath_IR = peaks_IR.copy()
valid_samples = np.arange(peaks_IR[0]-20, peaks_IR[0]+20, )
cleaned_impulseresp = np.zeros(impulse_resp.size)
cleaned_impulseresp[valid_samples] = impulse_resp[valid_samples]
cleaned_audio = convolve_linearchirp(cleaned_impulseresp,inverted_sweep )

plt.figure()
plt.subplot(311)
plt.plot(cleaned_impulseresp)
plt.plot(valid_samples, np.tile(0.01, valid_samples.size))
plt.subplot(312)
plt.specgram(cleaned_audio, Fs=fs, NFFT=256, noverlap=255)
plt.subplot(313)
plt.plot(cleaned_audio)
