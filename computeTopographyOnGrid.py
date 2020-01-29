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
#import matplotlib.pyplot as plt

def computeTopographyOnGrid(REFS, profile, opt, DDX, withWindow):
       h0 = opt[0]
       aC = opt[1]
       lC = opt[2]
       kC = opt[3]
       
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
              # Schar mountain (windowed with Kaiser bell)
              # Compute the height field
              ht1 = h0 * np.exp(-1.0 / aC**2.0 * np.power(xh, 2.0))
              ht2 = np.power(np.cos(mt.pi / lC * xh), 2.0)
              ht3 = np.reciprocal((1.0 / aC)**2.0 * np.power(xh, 2.0) + 1.0)
              if withWindow:
                     ht = kaiserDom * (ht1 * ht2 * ht3)
              else:
                     ht = 1.0 * (ht1 * ht2 * ht3)
              # Compute the slope field perfectly
              ht1 = h0 * np.exp(-1.0 / aC**2.0 * np.power(xh, 2.0))
              ht2 = np.power(np.cos(mt.pi / lC * xh), 2.0);
              ht = ht1 * ht2
              dht1 = -ht1
              dht2 = (2.0 / aC**2.0) * xh
              dht3 = ht2
              dht4 = (mt.pi / lC) * np.sin(2.0 * mt.pi / lC * xh)
              dhdx = dht2 * dht3
              dhdx = np.add(dhdx, dht4)
              dhdx = dht1 * dhdx
       elif profile == 3:
              # General even power exponential times a cosine series
              ht = np.zeros(len(xh))
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
       #return np.real(ht), np.real(dhdx)
       return ht, dhdx
              