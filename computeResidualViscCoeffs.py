#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import math as mt
import numpy as np
import bottleneck as bn
from scipy import ndimage

def computeResidualViscCoeffs(DIMS, state, RHS, RES, DLD, bdex, applyFilter):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       numVar = 1
       
       # Compute flow speed components
       UD = np.abs(state[:,0]); UD[UD < 1.0E-16] = 0.0
       WD = np.abs(state[:,1]); WD[WD < 1.0E-16] = 0.0
       # Compute flow speed along terrain
       VD = np.sqrt(bn.ss(state[:,0:2], axis=1))
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       RMAX = bn.nanmax(ARES,axis=0)
       RMAX[RMAX < 1.0E-16] = 1.0
       
       # Normalize each component residual and reduce to the measure on all variables
       QR = np.diag(np.reciprocal(RMAX))
       NARES = ARES @ QR
       CRES = bn.nanmax(NARES, axis=1)
       #CRES = bn.nanmean(NARES, axis=1) + bn.nanstd(NARES, axis=1)
       
       if applyFilter:
              nbrDex = DLD[-1] # List of lists of indices to regions
       
              # Apply the maximum filter over the precomputed regions
              CRESL = [bn.nanmax(CRES[reg]) for reg in nbrDex]
              CRES = np.array(CRESL)
       
       # Upper bound flow speed coefficients
       QMAX1 = 0.5 * DLD[0] * UD; #QMAX1[bdex] = 0.5 * DLD[0][bdex] * VD[bdex]
       QMAX2 = 0.5 * DLD[1] * WD; #QMAX2[bdex] = 0.5 * DLD[1] * VD[bdex]
       
       # Max norm of the limit coefficients
       QB1 = bn.nanmax(QMAX1)
       QB2 = bn.nanmax(QMAX2)
       
       # Residual based coefficients
       CRES1 = QB1 * CRES; CRES1[CRES1 < 1.0E-16] = 0.0
       CRES2 = QB2 * CRES; CRES2[CRES2 < 1.0E-16] = 0.0
       
       CRES1 = np.expand_dims(CRES1, axis=1)
       CRES2 = np.expand_dims(CRES2, axis=1)
       
       return (CRES1, CRES2)