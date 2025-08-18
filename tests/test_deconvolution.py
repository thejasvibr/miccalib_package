#!/usr/bin/env python

"""Tests for `miccalib` package."""


import unittest
from miccalib.deconvolution import deconvolve_linearchirp
from miccalib.deconvolution import convolve_linearchirp
from miccalib.deconvolution import eliminate_reflections_linchirp
from miccalib.utilities import dB, RMS, powerspec
import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt
import soundfile as sf

class TestMiccalib(unittest.TestCase):
    """Tests for `miccalib` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.fs = 192000 # Hz
        self.duration = 3e-3  # s 
        self.t = np.linspace(0, self.duration, int(self.fs*self.duration))
        self.f0, self.f1 = 95e3, 10e3 # Hz 
        self.synthetic_sweep = signal.chirp(self.t, self.f0,
                                            self.t[-1], self.f1,
                                            phi=np.pi/4)
        # define the 'core' bandwidth that doesn't include the 
        # windowed components
        self.coref0, self.coref1 = 93e3, 12e3 # Hz
        self.synthetic_sweep *= signal.windows.tukey(self.t.size, 0.1)
        self.synthetic_sweep *= 0.5
        
        self.sweep_rate = (self.f1-self.f0)/self.duration # Hz/s
        self.reverb_audio = np.zeros(self.t.size*10)
        # direct path
        self.t_direct = self.t.size
        self.reverb_audio[self.t_direct:self.t_direct+self.t.size] = self.synthetic_sweep
        self.only_directpath = self.reverb_audio.copy()
        self.relevant_samples = range(self.t_direct-50,
                                      self.t_direct+self.t.size+50)
        # indirect path 1 w delay of 1 ms
        self.t_indirect1 = self.t.size+int(self.fs*1e-3)
        self.reverb_audio[self.t_indirect1:self.t_indirect1+self.t.size] += self.synthetic_sweep*0.1
        
        # indirect path 2 w delay of 2 ms
        self.t_indirect2 = self.t_indirect1 + int(self.fs*1e-3)
        self.reverb_audio[self.t_indirect2 :self.t_indirect2 +self.t.size] += self.synthetic_sweep*0.05
        
        self.impulse_response, self.inverted_chirp = deconvolve_linearchirp(self.reverb_audio,
                                                                  self.sweep_rate,
                                                                  self.fs)
        # check that the peaks are at the expected delays 
        threshold = np.percentile(self.impulse_response, 99.5)
        self.peaks, self.props = signal.find_peaks(self.impulse_response, 
                                           threshold=threshold, 
                                           distance=int(self.fs*0.5e-3))
        
        self.expected = np.array([self.t_direct, self.t_indirect1, self.t_indirect2])
        self.observed = self.peaks[:3]
        
        # Clean up impulse respones
        mainpeak = self.peaks[0]
        relevant_range = np.arange(mainpeak-90, mainpeak+90)
        self.cleanedup_impresp = np.zeros(self.impulse_response.size)
        self.cleanedup_impresp[relevant_range] = self.impulse_response[relevant_range]
        
        # And convolve
        self.cleaned_audio = convolve_linearchirp(self.cleanedup_impresp, self.inverted_chirp)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        #plt.close()
        
    def test_timeofarrival_deconvolution(self):
        """Check that the impulse response peaks match the expected TOA"""
        expected_ends = self.expected + int(self.fs*self.duration)
        self.assertTrue(np.array_equal(expected_ends, self.observed))

    def test_dBrms_postconvolution(self):
        '''Clean up the impulse response and then check if the dBrms matches the original signal'''
        # plt.figure()
        # plt.plot(self.cleanedup_impresp)
        expected_dbrms = dB(RMS(self.only_directpath[self.relevant_samples]))
        observed_dbrms = dB(RMS(self.cleaned_audio[self.relevant_samples]))
        print(expected_dbrms, observed_dbrms)
        dbrms_difference = abs(observed_dbrms-expected_dbrms)
        self.assertTrue(dbrms_difference<0.05)
    
    def test_plots(self):
        cleanedaudio = self.cleaned_audio#[self.relevant_samples]
        directpath = self.only_directpath#[self.relevant_samples]
        cc = signal.correlate(cleanedaudio,
                              directpath, mode='same')
        peak = np.argmax(cc) - cleanedaudio.size*0.5
        plt.figure()
        plt.plot(cc)
        plt.vlines(cc.size*0.5, 0, np.max(cc),linestyles='dashed', color=['red'])
        
        print(f'Delay is: {peak}')
        plt.figure(figsize=(10,8))
        a0  = plt.subplot(211)
        plt.title('Comparing the simulated & de-convolved direct path sweep')
        plt.plot(cleanedaudio, label='reflections removed (direct path)')
        plt.plot(directpath, label='original simulated')
        plt.legend()
        plt.subplot(212, sharex=a0)
        plt.title('Difference in waveforms')
        diff_audio = self.cleaned_audio-self.only_directpath
        plt.plot(diff_audio)
        plt.savefig('original_convolved_comparison.png')
        
        sf.write('simulated_reverbaudio.wav',
                 self.reverb_audio,
                 self.fs)
        sf.write('deconvolved_directpath.wav',
                 self.cleaned_audio, 
                 self.fs)
        sf.write('simulated_groundtruth_onlydirectpath.wav',
                 self.only_directpath,
                 self.fs)
        
        #plt.savefig('original_convolved_comparison.png')
        
        
    def test_spectrum_postconvolution(self):
        '''how well does the convolved audio match the original'''
        direct_sweep = self.only_directpath[self.relevant_samples]
        conv_sweep = self.cleaned_audio[self.relevant_samples]
        plt.figure()
        plt.plot(conv_sweep)
        _, spectrum_direct = powerspec(direct_sweep,
                                           fs=self.fs)
        fft_freqs, spectrum_conv  = powerspec(conv_sweep,
                                             fs=self.fs)
        relevant_freqs = np.logical_and(fft_freqs>=self.coref1,
                                        fft_freqs<=self.coref0)
        
        # check the spectral power at the relevant frequencies
        spectralpower_diff = abs(spectrum_direct[relevant_freqs] - spectrum_conv[relevant_freqs])
        
        plt.figure()
        plt.plot(fft_freqs[relevant_freqs], spectralpower_diff)
        
        self.assertTrue(spectralpower_diff.max()<0.26) # dB

    def test_eliminate_reflections(self):
        '''Check that the high level function works'''
        
        without_reflections, (impresp, invsweep) = eliminate_reflections_linchirp(self.reverb_audio, 
                                                                         self.fs,
                                                                         {'freq_start':self.f0,
                                                                          'freq_stop':self.f1,
                                                                          'sweep_duration':self.duration
                                                                                 })
        diff = without_reflections[self.relevant_samples]-self.only_directpath[self.relevant_samples]
        # check that the main part of the diff signal is low. There will be an error 
        # in the edges for sure - but we need to see how to do this properly
        numsamples = without_reflections[self.relevant_samples].size
        numsamples_pct = numsamples-int(numsamples*0.95)       
        self.assertTrue(dB(RMS(diff))<0.05)
        
        
    # def test_command_line_interface(self):
    #     """Test the CLI."""
    #     runner = CliRunner()
    #     result = runner.invoke(cli.main)
    #     assert result.exit_code == 0
    #     assert 'miccalib.cli.main' in result.output
    #     help_result = runner.invoke(cli.main, ['--help'])
    #     assert help_result.exit_code == 0
    #     assert '--help  Show this message and exit.' in help_result.output
if __name__ == '__main__':
    unittest.main()