# -*- coding: utf-8 -*-
"""Tests for utilities module of the `miccalib` package."""


import unittest

import miccalib 
from miccalib.utilities import *
from miccalib.sensitivity import calc_native_freqwise_rms
import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt

class TestRMSFromFFT(unittest.TestCase):
    """Tests for `miccalib` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.fs = 192000 # Hz
        self.duration = 3e-3  # s 
        self.t = np.linspace(0, self.duration, int(self.fs*self.duration))
        self.f0, self.f1 = 80e3, 10e3 # Hz 
        self.synthetic_sweep = signal.chirp(self.t, self.f0,
                                            self.t[-1], self.f1,
                                            phi=np.pi/4)
       
    def tearDown(self):
        """Tear down test fixtures, if any."""
        
    def test_get_rms_from_fft(self):
        '''
        Check that the overall RMS calculated by 'direct' methods matches that calculated from 
        the FFT based RMS 
        '''
        overall_rms = rms(self.synthetic_sweep)
        fftfreqs, freqwise_rms = calc_native_freqwise_rms(self.synthetic_sweep, self.fs)
        # now calculate rms from FFT
        fftbased_rms = get_rms_from_fft(fftfreqs, freqwise_rms, freq_range=[0,self.fs*0.5])
        
        dbrms_difference = dB(fftbased_rms) - dB(overall_rms)
        self.assertTrue(dbrms_difference<=0.1)
        
        
    
    def test_raise_error_for_complex(self):
        '''
        Check that an error is raised if complex spectral power is input into the get_rms_from_fft
        '''
        complexfft = np.fft.fft(self.synthetic_sweep)
        fullfftfreqs = np.fft.fftfreq(complexfft.size, d=1/self.fs)
        self.assertRaises(ValueError, get_rms_from_fft,fullfftfreqs, complexfft,
                                                       freq_range=[0,self.fs*0.5])
        
        
        
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