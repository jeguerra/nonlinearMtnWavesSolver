#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:43:05 2019

@author: TempestGuerra
"""
import sys
import numpy as np
import math as mt
from scipy import signal
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def computeTopographyOnGrid(REFS, opt, DDX):
       h0 = opt[0]
       aC = opt[1]
       lC = opt[2]
       kC = opt[3]
       withWindow = opt[4]
       profile = opt[5]
       
       # Get data from REFS
       xh = REFS[0]
       NP = len(xh)
       
       # Make width for the Kaiser window
       r2 = 1.0 * kC
       r1 = -r2
                     
       # Make a window function so that dhdx = 0 inside Rayleigh layers
       condition1 = (xh > r1)
       condition2 = (xh < r2)
       condition = np.zeros(NP)
       
       for ii in range(NP):
              condition[ii] = condition1[ii] == 1 and condition2[ii] == 1
              
       WP = len(np.extract(condition, xh))
       kaiserWin = signal.kaiser(WP, beta=10.0)
       padP = NP - WP
       padZ = np.zeros(int(padP / 2))
       kaiserDom = np.concatenate((padZ, kaiserWin, padZ))
       #plt.figure()
       #plt.plot(x, kaiserDom)
       
       # Evaluate the function with different options
       if profile == 1:
              # Kaiser bell curve
              ht = h0 * kaiserDom
              # Take the derivative
              dhdx = DDX.dot(ht)
       elif profile == 2:
              # Schar mountain with analytical slope
              ht1 = h0 * np.exp(-1.0 / aC**2.0 * np.power(xh, 2.0))
              ht2 = np.power(np.cos(mt.pi / lC * xh), 2.0);
              ht = ht1 * ht2
              dhdx = -2.0 * ht
              dhdx *= (1.0 / aC**2.0) * xh + (mt.pi / lC) * np.tan(mt.pi / lC * xh)
       elif profile == 3:
              # General Kaiser window times a cosine series
              ps = 2.0 # polynomial order of cosine factor
              hs = 0.5 # relative height of cosine wave part
              hf = 1.0 / (1.0 + hs) # scale for composite profile to have h = 1
              ht2 = 1.0 + hs * np.power(np.cos(mt.pi / lC * xh), ps);
              ht = hf * h0 * kaiserDom * ht2
              ht[0] = 0.0; ht[-1] = 0.0
              # Take the derivative (DO NOT USE NATIVE DERIVATIVE OPERATOR)
              #dhdx_native = DDX.dot(ht)
              cs = CubicSpline(xh, ht, bc_type='periodic')
              dhdx = (cs.derivative())(xh)[:]
              # Monotonic filter
              dhdx[0] = 0.0; dhdx[-1] = 0.0
              for dd in range(1,len(dhdx)-1):
                     if ht[dd] == 0.0 and ht[dd+1] == 0.0 and ht[dd-1] == 0:
                            dhdx[dd] = 0.0
              #print(dhdx)
              '''
              plt.plot(xh, dhdx_native, xh, dhdx_cubic)
              plt.xlim(-25000, 25000)
              plt.figure()
              plt.plot(xh, dhdx_native - dhdx_cubic)
              plt.xlim(-25000, 25000)
              plt.show()
              print(ht)
              #print(dhdx_native)
              print(dhdx_cubic)
              input(
              '''
       elif profile == 4:
              # General even power exponential times a polynomial series
              ht = np.zeros(len(xh))
       elif profile == 5:
              # Terrain data input from a file, maximum elevation set in opt[0]
              ht = np.zeros(len(xh))
       else:
              print('ERROR: invalid terrain option.')
              sys.exit(2)
       '''       
       # Compute derivative by FFT
       if NP % 2 == 0:
              posRange = list(range(0, int(NP / 2)))
              negRange = list(range(-int(NP / 2 + 1), 0))
              k = np.array(posRange + [0] + negRange, dtype=np.float64)
       else:
              posRange = list(range(0, int((NP - 1) / 2)))
              negRange = list(range(-int((NP - 1) / 2), 0))
              k = np.array(posRange + [0] + negRange, dtype=np.float64)
              
       # Scale the frequency array
       ks = 2 * np.pi / (l2 - l1) * k
       # Compute derivative by FFT
       HF = np.fft.fft(htfft)
       DHDX = 1j * np.multiply(ks, HF)
       # Compute the orthogonal projection to the xh grid
       FIM = 1j * np.zeros((len(xh), NP))
       # Shift domain to positive
       xh += l2
       # Compute the Fourier basis on the desired grid
       for cc in range(len(k)):
              arg = 1j * ks[cc] * xh
              FIM[:,cc] = 1.0 / NP * np.exp(arg)
       xh -= l2
              
       # Compute the inverse Fourier interpolation
       ht = np.dot(FIM, HF)
       dhdx = np.dot(FIM, DHDX)
       '''
       
       # Fix h and dhdx to be zero at both ends
       ht[0] = 0.0; dhdx[0] = 0.0
       ht[-1] = 0.0; dhdx[-1] = 0.0
       #return np.real(ht), np.real(dhdx)
       return ht, dhdx
              