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
from scipy.interpolate import CubicSpline
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
       
       # Get a derivative operator
       #DDX, D2DX = derv.computeCubicSplineDerivativeMatrix(xh, False, False, None)
       DDXC = derv.computeCompactFiniteDiffDerivativeMatrix1(xh, 6)
       DDX, DDX4 = derv.computeQuinticSplineDerivativeMatrix(xh, np.zeros(DDXC.shape))
       D2DX = DDX @ DDX
       
       # Make width for the Kaiser window
       r2 = 1.0 * kC
       r1 = -r2
       L = abs(r2 - r1)
       #'''
       sinDom = np.zeros(NP)
       sin2Dom = np.zeros(NP)
       cosDom = np.zeros(NP)
       ii = 0
       qs = 2.8
       for xx in xh:
              if xx <= r2 and xx >= r1:
                     sinDom[ii] = abs(mt.sin(mt.pi / L * xx))
                     sarg = 1.0 - sinDom[ii]**qs
                     sin2Dom[ii] = sarg**2.0
                     cosDom[ii] = mt.cos(mt.pi / L * xx)
              elif xx > r2:
                     sinDom[ii] = 1.0
                     sin2Dom[ii] = 0.0
                     cosDom[ii] = 0.0
              elif xx < r1:
                     sinDom[ii] = -1.0
                     sin2Dom[ii] = 0.0
                     cosDom[ii] = 0.0
                     
              ii += 1
       
       # Evaluate the function with different options
       if profile == 1:
              # Kaiser bell curve
              ht = h0 * sin2Dom
              # Take the derivative
              dhdx = DDX @ ht
              dhdx[np.abs(dhdx) < 1.0E-15] = 0.0
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
              hs = 0.75 # relative height of cosine wave part
              hf = 1.0 / (1.0 + hs) # scale for composite profile to have h = 1
              ht2 = 1.0 + hs * np.power(np.cos(mt.pi / lC * xh), ps)
              ht = hf * h0 * sin2Dom * ht2
              
              dhdx = DDX @ ht
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
       
       d2hdx2 = D2DX @ ht
       d2hdx2[np.abs(d2hdx2) < 1.0E-15] = 0.0
       
       S = np.power(1.0 + np.power(dhdx, 2.0), -0.5)
       S2 = np.reciprocal(1.0 + np.power(dhdx, 2.0))
       
       dSdx = DDX @ S
       
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
       
       return ht, dhdx, (d2hdx2, dSdx, S, S2)
              