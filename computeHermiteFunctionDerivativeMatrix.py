#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:10:03 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
from HerfunChebNodesWeights import hefuncm, hefunclb
#import matplotlib.pyplot as plt

def computeHermiteFunctionDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       L1 = DIMS[0]
       L2 = DIMS[1]
       NX = DIMS[3]
       
       alpha, whf = hefunclb(NX)
       HT = hefuncm(NX-1, alpha, True)
       HTD = hefuncm(NX, alpha, True)
       
       # Get the scale factor
       b = (np.amax(alpha) - np.min(alpha)) / abs(L2 - L1)
       
       # Make a diagonal matrix of weights
       W = np.diag(whf, k=0)
       
       # Compute the coefficients of spectral derivative in matrix form
       SDIFF = np.zeros((NX+1,NX));
       SDIFF[0,1] = mt.sqrt(0.5);
       SDIFF[NX,NX-1] = -mt.sqrt(NX * 0.5);
       SDIFF[NX-1,NX-2] = -mt.sqrt((NX - 1) * 0.5);

       for cc in range(NX-3,-1,-1):
              SDIFF[cc+1,cc+2] = mt.sqrt((cc + 2) * 0.5);
              SDIFF[cc+1,cc] = -mt.sqrt((cc + 1) * 0.5);
              
       # Hermite function spectral transform in matrix form
       STR_H = (HT).dot(W);
       # Hermite function spatial derivative based on spectral differentiation
       temp = (HTD.T).dot(SDIFF)
       temp = temp.dot(STR_H)
       DDM = b * temp
       
       return DDM, STR_H
       
       