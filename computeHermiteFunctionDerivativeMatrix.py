#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:10:03 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
from HerfunChebNodesWeights import hefuncm, hefunclb

def computeHermiteFunctionDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       L2 = DIMS[0]
       L1 = DIMS[1]
       NX = DIMS[3]
       
       alpha, whf = hefunclb(NX)
       HT = hefuncm(NX-1, alpha, True)
       HTD = hefuncm(NX, alpha, True)
       
       # Get the scale factor
       b = np.max(alpha) / (0.5 * (L2 - L1))
       
       # Make a diagonal matrix of weights
       W = np.diag(whf, k=0)
       
       # Compute the coefficients of spectral derivative in matrix form
       SDIFF = np.zeros((NX+1,NX));
       SDIFF[0,1] = mt.sqrt(0.5);
       SDIFF[NX,NX-1] = -mt.sqrt(NX * 0.5);
       SDIFF[NX-1,NX-2] = -mt.sqrt((NX - 1) * 0.5);

       for cc in range(NX-3,0,-1):
              SDIFF[cc,cc+1] = mt.sqrt((cc + 1) * 0.5);
              SDIFF[cc,cc-1] = -mt.sqrt(cc * 0.5);

       # Hermite function spectral transform in matrix form
       STR_H = np.matmul(HT, W);
       # Hermite function spatial derivative based on spectral differentiation
       DDM = b * np.matmul(np.matmul(HTD.T, SDIFF), STR_H);
       
       return DDM, STR_H
       
       