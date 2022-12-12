# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:03:02 2022

@author: z00664732

# f1    = input signal frequency
# fs    = sampling freqency
# Nfft  = number of fft
# prime = the complete sampling period
# (t, adcin) = mimic analog input signal

# bit   = number of bits in ADC
# ncomp = noise of the comparator
# ndac  = noise of the c-dac
# nsamp = sampling kT/C noise
# radix = radix of the C-DAC
# cu       = unit capacitance
# mismatch = capacitor mismatch in C-DAC

"""

import numpy as np
from sar import SAR
# from fft import fft_calc
# import matplotlib.pyplot as plt
# from operator import itemgetter


## Switches
plot = 1
save_fig = 1

## This simulates a standard X-bit ADC
bit   = 10
ncomp = 0#.01
ndac  = 0
nsamp = 0
radix = 2
cu       = 1 # unit:fF
mismatch = 0#.05 # [sigma~0.047fF]

## Define ADC
myadc = SAR(bit, ncomp, ndac, nsamp, radix, cu, mismatch)

## make an ideal sin signal
A = 10      # input sin amplitude
B = 1e-3    # input sin offset [make B samll]
f1 = 3e6    # input freq
fs = 100e6  # sample freq
prime = 101#41 # Nfft/fs = prime/f1
Nfft = 1024#np.power(radix, bit)#1024#512#128
fs = fs if fs/f1>2.56 else f1*2.56 # Nyquist
fs = f1 * Nfft/prime
## input signal generation
M = 10 #the ration of analog input signal and sampled signal, just for plotting
t = np.arange(0, Nfft/fs, 1/fs/M)
adcin = A*np.sin(2*np.pi*f1*t) + B #+ 0.001 * np.random.randn(N) 

## run adc and fft
(SNDR, ENOB) = myadc.adc_fft(t, adcin, fs, Nfft, prime, M, plot, save_fig)
