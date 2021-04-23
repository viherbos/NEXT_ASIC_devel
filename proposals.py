import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
from scipy import signal
from scipy.optimize import leastsq
from scipy.optimize import least_squares



class mosfet(object):
    def __init__(self,**kwargs):
        self.KPP_2n = kwargs['KPP_2n']
        self.W_L = kwargs['W_L']
        self.VTH = kwargs['VTH']

    def iDS(self,vGS):
        i_DS_aux = self.KPP_2n*self.W_L*(vGS-self.VTH)**2
        return i_DS_aux

    def vGS(self,iDS):
        v_GS_aux = np.sqrt(iDS/(self.KPP_2n*self.W_L))+self.VTH
        return v_GS_aux


class capacitor(object):
    def __init__(self, C, time_unit):
        self.C = C
        self.time_unit = time_unit

    def iC(self,vC):
        # A first zero must be introduced to compensate for the one sample loss
        return np.diff(vC) * self.C / self.time_unit

    def vC(self, iC):
        return np.cumsum(iC) * (1.0/self.C) * self.time_unit


class translinear_loop(object):
    # Functions are written taking into account the usage of refined MOSFET models
    # to study the effect of inversion level changes etc.

    def __init__(self, KPP_2n=150E-6, W_L=10, VTH=0.55):
        self.MN_1 = mosfet(KPP_2n=KPP_2n, W_L=W_L,   VTH=VTH)
        self.MN_2 = mosfet(KPP_2n=KPP_2n, W_L=W_L * 4, VTH=VTH)

    def VTL_mult_div(self,vGS_C,I_A,sqrt_term):
        vGS_A = self.MN_1.vGS(I_A)
        return (self.MN_1.iDS(vGS_C)+I_A)-(self.MN_2.iDS((vGS_A+vGS_C)/2.0)-sqrt_term)

    def geo_mean(self,I_A,I_B):
        # Translinear loop follows basic eq. vGS_o = vGS_A + vGS_B
        vGS_o = (self.MN_1.vGS(I_A) + self.MN_1.vGS(I_B))/2.0
        I_o   = self.MN_2.iDS(vGS_o)
        # Now we apply substraction of I_A and I_B currents
        I_C   = I_o - (I_A + I_B)
        return I_C

    def mult_div(self,I_X,I_Y,I_A):
        sqrt_term = self.geo_mean(I_X,I_Y)
        # Translinear loop follows basic eq. vGS_o = vGS_A + vGS_B
        # Adapted to feedback loop, solved with least_squares

        I_C = []
        for i in range(0,len(sqrt_term)):

            EFunc_Lambda = lambda vGS_C: self.VTL_mult_div( vGS_C, I_A[i], sqrt_term[i])

            vGS_C = least_squares( EFunc_Lambda,
                                    x0=0,
                                    bounds=(0,3.3),
                                    ftol=1e-10,
                                    xtol=1e-10,
                                    gtol=1e-15)

            I_C.append(self.MN_1.iDS(vGS_C['x']))

        return np.array(I_C).reshape(-1)


class SQRT_integrator(object):
    def __init__(self, KPP_2n=150E-6, W_L=10, VTH=0.55, time_unit=100E-12,
                 cap=10E-12, I_o=1E-6, I_a=1E-6):
        self.transl    = translinear_loop(KPP_2n, W_L, VTH)
        self.cap       = capacitor(cap,time_unit)
        #self.MN_3      = mosfet(KPP_2n=KPP_2n, W_L=W_L/2.0, VTH=VTH)
        self.MN_4      = mosfet(KPP_2n=KPP_2n, W_L=W_L, VTH=VTH)
        self.time_unit = time_unit
        self.vGS_C_prev = VTH
        self.iC_prev = 0
        self.I_o = I_o
        self.I_a = I_a

    def Efunc_integrator1(self, Iout, I_in):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        gmean      = self.transl.geo_mean(I_o, Iout)
        mult_div_1 = self.transl.mult_div(I_o + I_in, I_a, gmean/2.0)
        mult_div_2 = self.transl.mult_div(I_o, I_a, gmean/2.0)
        iC = mult_div_1 - mult_div_2 #iC = mult_div_1
        vGS_C  = self.cap.vC(iC) + self.vGS_C_prev
        Iout_est = self.MN_4.iDS(vGS_C)
        return Iout - Iout_est

    def Efunc_integrator2(self, Iout, I_in):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        gmean      = self.transl.geo_mean(I_o, Iout)
        mult_div_1 = self.transl.mult_div(I_o + I_in, I_a, gmean/2.0)
        mult_div_2 = self.transl.mult_div(I_o, I_a, gmean/2.0)
        iC = mult_div_1 - mult_div_2 #iC = mult_div_1
        vGS_C  = self.cap.vC(iC) + self.vGS_C_prev
        vGS_est = self.MN_4.vGS(Iout)
        return vGS_C - vGS_est


    def __call__(self, I_in):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        Iout_array=[]
        for i in range(0,len(I_in)):

            EFunc_Lambda = lambda Iout: self.Efunc_integrator1(Iout,
                                                              np.array([I_o[i]]),
                                                              np.array([I_a[i]]),
                                                              np.array([I_in[i]]))

            Iout = least_squares( EFunc_Lambda,
                                    x0=2E-9,
                                    bounds=(0,100E-6),
                                    ftol=5e-16,
                                    xtol=5e-23,
                                    gtol=5e-23,
                                    method='trf',
                                    verbose=1)

            Iout_array.append(Iout['x'])

            self.vGS_C_prev = self.MN_4.vGS(Iout['x'])

        #return np.cumsum(np.array(Iout_array).reshape(-1))
        return np.array(Iout_array).reshape(-1)


    def ideal(self, I_in, Gain=1):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        Iout = np.cumsum(I_in*Gain)*self.time_unit
        tau  = np.divide(self.cap.C * np.sqrt(I_o), np.sqrt(4*self.MN_4.KPP_2n*self.MN_4.W_L) * I_a)
        # Constant 4 comes from BETA definition as KPP_2n * 2
        Iout = np.divide(Iout,tau)
        vGS_C = self.MN_4.vGS(Iout)

        return vGS_C


class LIN_integrator(object):
    def __init__(self, KPP_2n=150E-6, W_L=10, VTH=0.55, time_unit=100E-12,
                 cap=10E-12, I_o=1E-6, I_a=1E-6):
        self.transl    = translinear_loop(KPP_2n, W_L, VTH)
        self.cap       = capacitor(cap,time_unit)
        #self.MN_3      = mosfet(KPP_2n=KPP_2n, W_L=W_L/2.0, VTH=VTH)
        self.MN_4      = mosfet(KPP_2n=KPP_2n, W_L=W_L, VTH=VTH)
        self.time_unit = time_unit
        self.vGS_C_prev = VTH
        self.iC_prev = 0
        self.I_o = I_o
        self.I_a = I_a

    def Efunc_integrator1(self, Iout, I_in):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        gmean      = self.transl.geo_mean(I_o, Iout)
        mult_div_1 = self.transl.mult_div(I_o + I_in, I_a, gmean/2.0)
        mult_div_2 = self.transl.mult_div(I_o, I_a, gmean/2.0)
        iC = mult_div_1 - mult_div_2 #iC = mult_div_1
        vGS_C  = self.cap.vC(iC) + self.vGS_C_prev
        Iout_est = self.MN_4.iDS(vGS_C)
        return Iout - Iout_est

    def Efunc_integrator2(self, Iout, I_in):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        gmean      = self.transl.geo_mean(I_o, Iout)
        mult_div_1 = self.transl.mult_div(I_o + I_in, I_a, gmean/2.0)
        mult_div_2 = self.transl.mult_div(I_o, I_a, gmean/2.0)
        iC = mult_div_1 - mult_div_2 #iC = mult_div_1
        vGS_C  = self.cap.vC(iC) + self.vGS_C_prev
        vGS_est = self.MN_4.vGS(Iout)
        return vGS_C - vGS_est


    def __call__(self, I_in):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        Iout_array=[]
        for i in range(0,len(I_in)):

            EFunc_Lambda = lambda Iout: self.Efunc_integrator1(Iout,
                                                              np.array([I_o[i]]),
                                                              np.array([I_a[i]]),
                                                              np.array([I_in[i]]))

            Iout = least_squares( EFunc_Lambda,
                                    x0=2E-9,
                                    bounds=(0,100E-6),
                                    ftol=5e-16,
                                    xtol=5e-23,
                                    gtol=5e-23,
                                    method='trf',
                                    verbose=1)

            Iout_array.append(Iout['x'])

            self.vGS_C_prev = self.MN_4.vGS(Iout['x'])

        #return np.cumsum(np.array(Iout_array).reshape(-1))
        return np.array(Iout_array).reshape(-1)


    def ideal(self, I_in, Gain=1):
        I_o = self.I_o*np.ones(len(I_in))
        I_a = self.I_a*np.ones(len(I_in))
        Iout = np.cumsum(I_in*Gain)*self.time_unit
        tau  = np.divide(self.cap.C * np.sqrt(I_o), np.sqrt(4*self.MN_4.KPP_2n*self.MN_4.W_L) * I_a)
        # Constant 4 comes from BETA definition as KPP_2n * 2
        Iout = np.divide(Iout,tau)
        return Iout



class LOG_integrator(object):
    def __init__(self, KPP_2n=150E-6, W_L=10, VTH=0.55, time_unit=100E-12,
                 cap1=10E-12, cap2=10E-12, R=10E3, Ibias1=1E-6, Ibias2=1E-6, Ibias3=1E-6):
        self.transl = translinear_loop(KPP_2n, W_L, VTH)
        self.cap1   = capacitor(cap1,time_unit)
        self.cap2   = capacitor(cap2,time_unit)
        self.R      = R
        self.time_unit = time_unit
        self.Ibias1 = Ibias1
        self.Ibias2 = Ibias2
        self.Ibias3 = Ibias3

    def __call__(self,Iin,Ibias1,Ibias2,Ibias3):
        Ibias1 = self.Ibias1*np.ones(len(Iin))
        Ibias2 = self.Ibias2*np.ones(len(Iin))
        Ibias3 = self.Ibias3*np.ones(len(Iin))

        V_C1 = self.cap1.vC(Iin)
        Im1  = self.transl.mult_div(Iin + Ibias1, Ibias2, V_C1/self.R + Ibias3)
        Im2  = self.transl.mult_div(Ibias1, Ibias2, V_C1/self.R + Ibias3)
        Im3  = Im1 - Im2
        return self.cap2.vC(Im3)

    def ideal(self, Iin, Gain=1):
        #K = Ibias2*self.R*(self.cap1.C/self.cap2.C)
        #return K*np.log(self.cap1.vC(Iin)/self.R+Ibias3)
        #return (Ibias2/self.cap2.C)*np.cumsum(np.divide(Iin,Ibias3+self.cap1.vC(Iin)/self.R))*self.time_unit
        Ibias1 = self.Ibias1*np.ones(len(Iin))
        Ibias2 = self.Ibias2*np.ones(len(Iin))
        Ibias3 = self.Ibias3*np.ones(len(Iin))

        return (Ibias2/self.cap2.C)*(self.R*self.cap1.C)*(np.log(self.cap1.vC(Iin*Gain)/self.R+Ibias3)-np.log(Ibias3))
