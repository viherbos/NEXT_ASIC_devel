import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from proposals import SQRT_integrator as integrator
from proposals import LOG_integrator
from waveforms import wave_gen


def find_nearest_bigger(A,val):
    return np.where(A-val >= 0, A-val, np.inf).argmin()

def find_nearest(A,val):
    return np.abs(A-val).argmin()

def PE_2_LSB_LUT(DAC_array, PE_charge_array, PE_inc, MAX_PE):
    # PE_inc = PE delta from position to position in array
    # Function returns PE array
    pe_lsb_lut=[]
    for code_v in DAC_array:
        data = find_nearest_bigger(PE_charge_array,code_v)
        pe_lsb_lut.append(data*PE_inc)
    PE_LSB_LUT = np.array(pe_lsb_lut)
    LSB_per_PE = np.array([np.sum(PE_LSB_LUT == x) for x in np.arange(PE_inc, MAX_PE,PE_inc)])
    return PE_LSB_LUT,LSB_per_PE


class evaluation(object):
    def __init__(self, time_unit, max_pe, lsb_pe, pulse_length,
                 signal, PE_array, PE_inc):
        # KPP_2n=150E-6, W_L=10, VTH=VTH, time_unit=100E-12, cap=10E-12
        self.tu       = time_unit
        self.max_pe   = max_pe
        self.lsb_pe   = lsb_pe
        self.p_length = pulse_length
        self.Ipeak_array   = []
        self.vGSpeak_array = []
        self.signal = signal
        self.PE_array = PE_array
        self.PE_inc = PE_inc

        # Bias currents
        # self.I_o    = bias['I_o']*np.ones(self.p_length)
        # self.I_a    = bias['I_a']*np.ones(self.p_length)
        # self.Ibias1 = bias['Ibias1']*np.ones(self.p_length)
        # self.Ibias2 = bias['Ibias2']*np.ones(self.p_length)
        # self.Ibias3 = bias['Ibias3']*np.ones(self.p_length)


    def get_integral(self, I, Gain):
        out = []
        for i in range(1,len(self.PE_array)):
            out_aux = I.ideal(self.signal[i*self.p_length:(i+1)*self.p_length], Gain=Gain)
            out.append(np.max(out_aux))
        return np.array(out)


    def lsb_2_pe(self, I, Nbits, LSB, Gain, offset_corr=0):
        Iout = self.get_integral(I, Gain)
        range  = np.array([x for x in np.arange(0,2**Nbits)])*LSB
        outA, dump = PE_2_LSB_LUT(range, Iout-offset_corr, self.PE_inc, self.max_pe)
        aux = np.divide(np.array([x for x in np.arange(0,2**Nbits)]),outA)
        return outA, aux


    def res_compute(self, PE_LSB_LUT, LSB_per_PE):
        RES = np.divide(np.divide(np.ones(len(LSB_per_PE)),LSB_per_PE),PE_LSB_LUT)*100
        return RES
