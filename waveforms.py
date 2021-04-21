import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
from antea.elec import tof_functions as wvf
from scipy import signal


# Waveform generator
def wave_gen(pe_time_zs:np.array, time_unit, Bandwidth) -> Tuple[np.array,np.array]:

    # Constants
    q = 1.6021E-19
    #Electron charge
    SIPM_gain = 500000
    # Time unit (tu) -> 100 ps
    # Simulation Time Step -> 5ns (50 tu)

    spe_response,norm = wvf.apply_spe_dist(np.arange(0,20000),[10,500])
    spe_response_C = spe_response * SIPM_gain * q
    #spe_response in Coulombs
    spe_response_A = spe_response_C / (time_unit)
    #spe_response in Amperes

    print("CHECK: Electrons in spe_response = %f" % (np.sum(spe_response_A)*time_unit/q/SIPM_gain))

    time = np.arange(0,pe_time_zs[0,-1].astype('int')+len(spe_response_A))
    pe   = np.zeros(pe_time_zs[0,-1].astype('int')+len(spe_response_A))
    pe[pe_time_zs[0,:].astype('int')] = pe_time_zs[1,:]

    # C. Romo convolution
    wave = wvf.convolve_tof(spe_response_A,pe)

    # Shaping based on bandwidth guess
    f_sample = (1/time_unit); # Hz
    BW_guess = Bandwidth; # Hz
    #freq_LPF = 1/(100*1E-9); # rad/sec
    freq_LPF = 2*np.pi*BW_guess; # rad/sec
    freq_LPFd = freq_LPF / (f_sample*np.pi); # Normalized by Nyquist Freq (half-cycles/sample)
    # Filter Definitions
    b, a = signal.butter(1, freq_LPFd, 'low', analog=False)
    signal_out = signal.lfilter(b,a,wave)

    return time,signal_out
