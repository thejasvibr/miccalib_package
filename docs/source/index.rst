.. miccalib documentation master file, created by
   sphinx-quickstart on Sun May  4 17:02:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to miccalib's documentation!
====================================
This is an under-development package to implement substitution-type microphone calibration. The brief idea behind substitution-type calibration is that you have a mic with a known frequency response (the 'calibration' mic), and a target mic whose frequency response you'd like to calibrate (the 'target' mic). By using a common playback signal and placing both mics in the same location w.r.t to the speaker - you are now capturing the 'same' sound on both mics. Using the calibration mic, you can calculate the playback's sound levels (:math:`\frac{Pa_{rms}}{Hz}`) and thus calculate your target mic's sensitivity (:math:`\frac{a.u.rms}{Pa}` or :math:`\frac{V_{rms}}{Pa}`).

To use this package your data will need to be collected in a particular way (as of August 2025):
	* A target microphone that you'd like to characterise
	* A calibration microphone with a flat frequency response (implemented) e.g. GRAS 40BF or the like) OR  a microphone whose frequency response has been previously characterised (not yet implemented)
	* A couple of tripods
	* A meter stick, or laser-range finder
	* A couple of tripods 
	* A speaker that produces consistent playbacks (Very very important)
	* A digital playback sweep file to playback on the speaker (or also use the sweep provided)

The 'examples' section shows you what can be done after the data has been collected. The experimental protocol used for the data in the 'examples' section is described in de Framond et al. 2025 Curr. Biol. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
