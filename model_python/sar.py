# -*- coding: utf-8 -*-
import copy
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

"""
sar adc model

t       = time stamp of input signal
adcin   = input signal of the ADC 
fs      = sample frequency
Nfft    = number of sample points for FFT
prime   = the number of complete cycles of input
M       = the ration of analog input signal and sampled signal, just for plotting

bit     = number of bits in SAR ADC
ncomp   = noise of the comparator
ndac    = noise of the c-dac
nsamp   = sampling kT/C noise
radix   = radix of the C-DAC
cu      = unit capacitance
mismatch= capacitor mismatch in C-DAC
"""

class SAR:
    def __init__(self, bit, ncomp, ndac, nsamp, radix, cu=5, mismatch=0.01):
        self.ncomp  = ncomp
        self.ndac   = ndac
        self.nsamp  = nsamp
        self.bit    = bit
        self.radix  = radix
        self.cu     = cu
        
        print("Simulating a {} bit SAR ADC".format(bit))
        
        if mismatch:
            print("Adding capacitor mismatch: %.1f%%" % (mismatch*100))
            self.mismatch = mismatch # default 1% mismatch for unit cap
        else:
            print("Capacitor mismatch not included")
            self.mismatch = 0
        
        # # create CDAC
        # self.cdac = self.dac()
            
    def sample(self, t, adcin, fs, Nfft):
        adcin_dict = dict(zip(np.round(t,10), adcin))
        ts = np.arange(0, Nfft/fs, 1/fs) #sampling time
        adcs_tuple = itemgetter(*np.round(ts,10))(adcin_dict)
        adcs = np.array(adcs_tuple) #sampling value
        #add sampling noise
        adcs += np.random.randn(adcs.shape[0]) * self.nsamp
        return ts, adcs
            
    def dac(self):
        cdac = np.zeros((self.bit,1))
        for i in range(self.bit):
            cdac[i] = self.cu * np.power(self.radix,(self.bit-1-i))
            # add mismatch 
            mis = self.mismatch * 3 # 3sigma
            cdac[i] += np.random.randn()*mis
        
        #normalize to full scale = 1
        cdac = cdac/(sum(cdac)+(self.cu + np.random.randn()*mis)) #dnt forget the last unit cap!
        return cdac
    
    def comp(self, compin):
        #note that comparator output suffers from noise of comp and dac
        comptemp = compin + np.random.randn(compin.shape[0])*self.ncomp + np.random.randn(compin.shape[0])*self.ndac
        
        #comp function in vectors
        out = np.sign(comptemp) # out=1/-1
        return out

    def conv(self, t, adcin, fs, Nfft):
        adcin = copy.deepcopy(adcin) #deep copy input to avoid overwrite
        _,adcs = self.sample(t, adcin, fs, Nfft)
        adcout = np.zeros_like(adcs) #initialize adcout as a list of zreos with the same number of adcs
        ref = np.max(adcs)
        #loop for sar cycles in each conversion
        for cyloop in range(self.bit):
            compout = self.comp(adcs)
            adcs += compout * (-1) * self.dac()[cyloop] * ref #update cdac output
            adcout += np.power(self.radix, self.bit-1-cyloop)*np.maximum(compout, 0) # accumulate only when compout is 1
        # #normalize
        adcout /= np.power(self.radix, self.bit)
        #shift and scale for comparison with dacin
        adcout = (adcout*2-1)*ref
        return adcout
    
    def adc_fft(self, t, adcin, fs, Nfft, prime, M, plot=False, save=False):
        # inputs sampled analog waveform with sample rate 
        # returns SNDR + ENOB
        adcout = self.conv(t, adcin, fs, Nfft)
        
        # FFT
        F = np.fft.fft(adcout)

        # Spectrum analyze
        N = Nfft
        Amp = np.power(np.abs(F)[0:int(N/2)], 2)
        Amp[0] = 0 # cut DC
        # normalize
        Amp /= max(Amp[1:])
        # db
        Amp_db = 10*np.log10(Amp)
        
        freq = np.arange(0, fs/2, fs/N)

        ts = np.arange(0, N/fs, 1/fs) # time
        # sampled value
        ts, adcs = self.sample(t, adcin, fs, Nfft)
        ## Normalize your input
        norm_adcin, center, maxbin = normalize_input(adcin)
        norm_adcs, _, _ = normalize_input(adcs)
        ## Rescale ADC output to original
        norm_adcout, _ , _ = normalize_input(adcout)
        
        if plot:  # plot signal waveforms
            print("plotting conversion results")
            plt.figure(figsize=(6,6))
            plt.subplot(211)
            plt.plot(t, adcin, label='adcin')
            plt.plot(ts, adcs, label='adcs')
            plt.plot(ts, adcout, label='adcout')
            # plt.plot(t[:int(Nfft/prime)*M], adcin[:int(Nfft/prime)*M], label='adcin')
            # plt.plot(ts[:int(Nfft/prime)+1], adcs[:int(Nfft/prime)+1], label='adcs')
            # plt.plot(ts[:int(Nfft/prime)+1], adcout[:int(Nfft/prime)+1], label='adcout')
            plt.legend(loc="upper right")
            plt.title('Original input/sample/output signals')
            plt.xlabel('Time [s]', fontsize=18)
            plt.ylabel('Amplitude [Unit]', fontsize=18)
            # plt.show()
            plt.subplot(212)
            plt.plot(t[:int(Nfft/prime)*M], norm_adcin[:int(Nfft/prime)*M], label='adcin')
            plt.plot(ts[:int(Nfft/prime)+1], norm_adcs[:int(Nfft/prime)+1], '-o', label='adcs')
            plt.plot(ts[:int(Nfft/prime)+1], norm_adcout[:int(Nfft/prime)+1], '-x', label='adcout')
            plt.legend(loc="upper right")
            plt.title('Normalized input/sample/output signals \n in one cycle')
            plt.xlabel('Time [s]', fontsize=18)
            plt.ylabel('Amplitude [Unit]', fontsize=18)
            
            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
                wspace=None, hspace=1)
            # plt.show()
        
       
        if plot:  # plot spectrum
            plt.figure()
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 17

            plt.plot(freq,Amp_db, label='|Amp(dB)|')
            plt.xlabel('Frequency [Hz]', fontsize=20)
            plt.ylabel('Amplitude [dB]', fontsize=20)
            plt.title('FFT result')
            plt.grid()
        if save:
            plt.savefig('FFT.png', bbox_inches='tight', dpi=150, transparent=True)
        
        plt.show()
        # SNR calc
        sig_bin = np.where(Amp==np.abs(Amp).max())[0]
        signal_power = Amp[sig_bin]

        noise_power = Amp.sum() - signal_power

        SNDR = signal_power / noise_power
        SNDR = 10*np.log10(SNDR)

        ENOB = (SNDR-1.76) / 6.02
        print("SNDR:", SNDR)
        print("ENOB:", ENOB)
        return SNDR, ENOB

def normalize_input(inp):
    # center = np.mean(inp)
    center = (np.max(inp) + np.min(inp)) / 2
    out = inp - center
    
    maxbin = np.max(out) #* 2
    out = out / maxbin
    
    return out, center, maxbin
    
    
if "__name__" == "main":
    # lets test the adc by random vectors..
    adcin = (np.random.rand(10000)-0.5)/2
    print(adcin)
    adc = SAR(10, 0, 0, 0, 2)
    adcout=adc.conv(adcin)
    
    #print it.
    adc.adc_fft(adcin)
