#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:43:05 2019

@author: TempestGuerra
"""
import sys
import numpy as np
import math as mt
import computeDerivativeMatrix as derv
from scipy import signal
import matplotlib.pyplot as plt

def computeTopographyOnGrid(REFS, opt):
       h0 = opt[0]
       aC = opt[1]
       lC = opt[2]
       kC = opt[3]
       profile = opt[5]
       
       # Get data from REFS
       xh = REFS[0]
       NP = len(xh)
       
       # Make width for the Kaiser window
       r2 = 1.0 * kC
       r1 = -r2
       L = abs(r2 - r1)
       
       sinDom = np.zeros(NP)
       sin2Dom = np.zeros(NP)
       cosDom = np.zeros(NP)
       ii = 0
       for xx in xh:
              if xx < r2 and xx > r1:
                     sinDom[ii] = mt.sin(mt.pi / L * xx)
                     sin2Dom[ii] = 1.0 - np.power(sinDom[ii], 2.0)
                     cosDom[ii] = mt.cos(mt.pi / L * xx)
              else:
                     sinDom[ii] = 0.0
                     sin2Dom[ii] = 0.0
                     cosDom[ii] = 0.0
              ii += 1
       
       # Evaluate the function with different options
       if profile == 1:
              # Kaiser bell curve
              ht = h0 * sin2Dom
              # Take the derivative
              dhdx = -(2.0 * mt.pi / L) * h0 * sinDom * cosDom
       elif profile == 2:
              # Schar mountain with analytical slope
              ht1 = h0 * np.exp(-1.0 / aC**2.0 * np.power(xh, 2.0))
              ht2 = np.power(np.cos(mt.pi / lC * xh), 2.0);
              ht = ht1 * ht2
              # Take the derivative
              dhdx = -2.0 * ht
              dhdx *= (1.0 / aC**2.0) * xh + (mt.pi / lC) * np.tan(mt.pi / lC * xh)
              
              ht[np.abs(ht) < 1.0E-15] = 0.0
              dhdx[np.abs(dhdx) < 1.0E-15] = 0.0
       elif profile == 3:
              # General Kaiser window times a cosine series
              ps = 2.0 # polynomial order of cosine factor
              hs = 0.5 # relative height of cosine wave part
              hf = 1.0 / (1.0 + hs) # scale for composite profile to have h = 1
              ht2 = 1.0 + hs * np.power(np.cos(mt.pi / lC * xh), ps)
              ht = hf * h0 * sin2Dom * ht2
              
              # Take the derivative
              '''
              dhdx1 = -(2.0 * mt.pi / L) * sinDom * cosDom
              dhdx2 = -hs * (ps * mt.pi / lC) * np.cos(mt.pi / lC * xh) * np.sin(mt.pi / lC * xh)
              dhdx = hf * h0 * (dhdx1 * ht2 + sin2Dom * dhdx2)
              '''
              DDX, temp = derv.computeCubicSplineDerivativeMatrix(xh, False, True, False, False, None)
              dhdx = DDX.dot(ht)
              dhdx[np.abs(dhdx) < 1.0E-15] = 0.0
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
       
       '''
       fc = 1.25
       plt.figure()
       plt.plot(xh, ht, 'k', linewidth=2.0)
       plt.xlim(-fc*kC, fc*kC)
       plt.figure()
       plt.plot(xh, dhdx, 'k', linewidth=2.0)
       plt.xlim(-fc*kC, fc*kC)
       plt.show()
       input()
       '''
       return ht, dhdx
              