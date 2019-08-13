#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:43:05 2019

@author: TempestGuerra
"""
import sys
import numpy as np
from numpy import multiply as mul
import math as mt
import matplotlib.pyplot as plt 

def computeTopographyOnGrid(REFS, profile, opt):
       
       # Get data from REFS
       xh = REFS[0]
       l2 = np.amax(xh)
       l1 = np.amin(xh)
       DX = 50.0 # maximum resolution...
       NP = int((l2 - l1) / DX)
       x = np.linspace(l1, l2, num=NP, endpoint=False)
       
       # Evaluate the function with different options
       if profile == 1:
              # Witch of Agnesi here
              h0 = opt[0]
              aC = opt[1]
       elif profile == 2:
              # Schar mountain
              h0 = opt[0]
              aC = opt[1]
              lC = opt[2]
              # Compute the height field
              ht1 = h0 * np.exp(-1.0 / aC**2.0 * np.power(x, 2.0))
              ht2 = np.power(np.cos(mt.pi / lC * x), 2.0);
              ht = mul(ht1, ht2)
              # Compute the slope field perfectly
              dht1 = -ht1
              dht2 = (2.0 / aC**2.0) * x
              dht3 = ht2
              dht4 = (mt.pi / lC) * np.sin(2.0 * mt.pi / lC * x)
              dhdx = mul(dht2, dht3)
              dhdx = np.add(dhdx, dht4)
              dhdx = mul(dht1, dhdx)
       elif profile == 3:
              # General even power exponential times a cosine series
              h0 = opt[0]
              ht = np.zeros(len(x))
       elif profile == 4:
              # General even power exponential times a polynomial series
              h0 = opt[0]
              ht = np.zeros(len(x))
       elif profile == 5:
              # Terrain data input from a file, maximum elevation set in opt[0]
              h0 = opt[0]
              ht = np.zeros(len(x))
       else:
              print('ERROR: invalid terrain option.')
              sys.exit(2)
              
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
       HF = np.fft.fft(ht)
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
       
       return np.real(ht), np.real(dhdx)
              