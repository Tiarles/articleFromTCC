
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:40:24 2019

@author: tiarl
"""

import scipy
from scipy import stats
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

#from numpy import pi
#from numpy.fft import fft, fftfreq, fftshift

#def w(x):
#    return np.abs(x)
#
#def fscmFunction(fmin, fmax, sf=12):
#    
#    B = fmax - fmin
#    T = 1/B
#    Ts = 2**sf * T
#    
#    n = 1000000
#    
#    h = np.arange(0, sf-1)
#    s = w(n*Ts) * 2**h
#    
#    return h, s
#
#h, s = fscmFunction(100, 500)
#
#plt.plot(h, s, 'ko-')
#plt.show()

#def fscmFunction2(fmin, fmax, sf=12):
#    import numpy as np
#    
#    #B = fmax - fmin
##    T = 1/B
##    Ts = 2**sf * T
#    
#    
#    k_array = np.arange(0, 2**(sf - 1))
#    
#    c1 = 1/np.sqrt(2**sf)
#    c2 = 1j*2*np.pi
#    _, c3 = fscmFunction(fmin, fmax, sf=sf)
#    c4 = 2**sf
#    
#    c = [c1*np.e**(c2*((c3 + k) % c4) * k/c4) for k in k_array]
#    
#    c_all = np.concatenate(c)
#    
#    c_time = ifft(c_all)
#    
#    return c_time
#
#c = fscmFunction2(100, 500)
#plt.plot(c[0:int(c.size/2 + 1)])
#plt.xlim(0,1000)
#print(c)

#dim_t = 2048
#err = 0.05
#Ts = 1e-6 / 2
#t = np.arange(0, 1, Ts)
#doppler = np.sqrt(t*(1-t)) * np.sin(2*pi*(1+err)/(t+err))
#doppler1 = np.sin(2*pi*(1+err)/(t+err))
#
#plt.plot(t, doppler1)
#plt.show()
#
#Y_par = fft(sinal)
#
#sinal = doppler1
#
#plt.plot(Y_par.imag)
#plt.xlim(0, 100)
#plt.show()
#
#plt.plot(Y_par.real)
#plt.xlim(0, 500)
#plt.show()



#Sinal = 2*np.abs(fft(sinal))[:int(sinal.size/2)] / sinal.size
#Sinal[0] = Sinal[0]/2 
#
#f = fftfreq(sinal.size, Ts)[:int(sinal.size/2)]
#plt.plot(f, Sinal, 'ko-')
#xmin, xmax = plt.xlim()
##plt.xlim(0, 100000)
#plt.show()

   
#t = np.linspace(0, 5, 2048)
#f = np.linspace(1, 20, 2048)
#
#s = np.e**(2*np.pi*f*t*1j)
#
#plt.plot(t, s)
#plt.xlim(0, 2)
#plt.show()

uplink_table = [
        (916.8*1e6, list(range(7, 11)), [125*1e3]), 
        (917.0*1e6, list(range(7, 11)), [125*1e3]),
        (917.2*1e6, list(range(7, 11)), [125*1e3]),
        (917.4*1e6, list(range(7, 11)), [125*1e3]),
        (917.6*1e6, list(range(7, 11)), [125*1e3]),
        (917.8*1e6, list(range(7, 11)), [125*1e3]),
        (918.0*1e6, list(range(7, 11)), [125*1e3]),
        (918.2*1e6, list(range(7, 11)), [125*1e3]),
        (918.5*1e6, [8], [500*1e3])]

print("")
print('    fmin e fmax para o canal 0 da AU915-928')
channel = uplink_table[0]
fcenter = channel[0]
bw = channel[2][0]
fmin, fmax = (fcenter - bw/2, fcenter + bw/2)

Ts = 1/(fmax * 10)
chip_Ts = 1e-3 # = 1 ms

print("fmin =", fmin/1e6,"MHz")
print("fmax =", fmax/1e6,"MHz")

value = int(input("Value: "))

N = 2**7

step = bw / N

fstart = fmin + step * value

import matplotlib.pyplot as plt

t = np.arange(0, chip_Ts, chip_Ts/N)
y = []
for i, ti in enumerate(t):
    next_y = fstart + step * i
    if next_y >= fmax:
        y.append(next_y - fmax + fmin)
    else:
        y.append(next_y)

y = np.array(y)

nCycles = 100000

y3 = np.zeros(y.size*nCycles)
for i in range(nCycles):
    y3[i::nCycles] = y


plt.hlines(fmax, t[0], t[-1], color='red', linestyle='--')
plt.hlines(fcenter, t[0], t[-1], color='green', linestyle='-.')
plt.hlines(fmin, t[0], t[-1], color='red', linestyle='--')
plt.plot(t, y, 'ko ')
plt.show()


t3 = np.arange(0, chip_Ts, chip_Ts/N/nCycles)
#t3 = t3[:-1]

y2 = np.sin(2*np.pi*y3*t3)
plt.plot(t3, y2, 'k-')
plt.xlim(0, 10/(fmax))
plt.show()

#y2 = np.sin(2*np.pi*y*t)
plt.plot(y3, 'k-')
plt.xlim(0, 100)
plt.show()