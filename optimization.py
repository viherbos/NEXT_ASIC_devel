import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from proposals import SQRT_integrator as integrator
from proposals import LOG_integrator
from waveforms import wave_gen


class linear_ADC(object):
    def __init__(self, mosfet_params, time_unit, cap, max_pe, lsb_pe, pulse_length,
                 signal, PE_array, PE_inc):
        # KPP_2n=150E-6, W_L=10, VTH=VTH, time_unit=100E-12, cap=10E-12
        self.tu       = time_unit
        self.cap      = cap
        self.max_pe   = max_pe
        self.lsb_pe   = lsb_pe
        self.p_length = pulse_length
        self.I1 = integrator(KPP_2n = mosfet_params['KPP_2n'],
                                W_L = mosfet_params['W_L'],
                                VTH = mosfet_params['W_L'],
                                time_unit = self.tu,
                                cap = self.cap)
        self.Ipeak_array   = []
        self.vGSpeak_array = []
        self.signal = signal
        self.PE_array = PE_array
        self.PE_inc = PE_inc


    def get_integral(self):
        for i in range(1,len(self.PE_array)):
            Iout,vGS_C = self.I1.ideal(I_o, I_a, self.signal[i*self.p_length:(i+1)*self.p_length])
            self.Ipeak_array.append(np.max(Iout))
            #self.vGSpeak_array.append(np.max(vGS_C))

        self.Ipeak_array_L = np.array(self.Ipeak_array)
        #self.vGSpeak_array_L = np.array(self.vGSpeak_array)

        return MAX_current = np.max(self.Ipeak_array_L)
