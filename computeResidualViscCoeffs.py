#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn
from scipy import ndimage
from numba import njit, prange

@njit(parallel=True)
def computeVariableNorm(ARES, LVAR):
       
       Q_RES = np.empty(LVAR)
       for ii in prange(LVAR):
              Q_RES[ii] = ARES[ii,:].sum()
              
       return Q_RES

@njit(parallel=True)
def computeRegionFilterBound(UD, WD, Q_RES, DLD, nbrDex, LVAR):
              
       Q1 = np.empty((LVAR,1))
       Q2 = np.empty((LVAR,1))

       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              
              # Compute the local L-inf norm in a region
              UDM = UD[rdex].max()
              WDM = WD[rdex].max()
              QDM = Q_RES[rdex].max()
              
              # Compute the minimum bound on coefficients
              Q1[ii,0] = min(0.5 * DLD[0] * UDM, DLD[2] * QDM)
              Q2[ii,0] = min(0.5 * DLD[1] * WDM, DLD[3] * QDM)
              
       return Q1, Q2

def computeResidualViscCoeffs2(DIMS, AV, MAG, DLD, bdex, ldex, applyFilter, RLM, DCFC):
       
       # Get the region indices map
       nbrDex = DLD[-1] # List of lists of indices to regions
       
       # Compute flow speed along terrain
       UD = AV[:,0]# + DCFC[1]
       WD = AV[:,1]# + DCFC[1]
       
       # Compute absolute value of residuals
       AMAG = np.abs(MAG)
       LVAR = MAG.shape[0]
       
       # Reduce across the variables using the 1-norm
       Q_RES = bn.nansum(AMAG, axis=1)
       
       #%% Filter to spatial regions and apply stability bounds
       CRES1, CRES2 = computeRegionFilterBound(UD, WD, Q_RES, DLD, nbrDex, LVAR)
       
       #%% SET DAMPING WITHIN THE ABSORPTION LAYERS
       CRES1[ldex,0] += DCFC[0] * RLM[0,ldex]
       CRES2[ldex,0] += DCFC[0] * RLM[0,ldex]
       
       return (CRES1, CRES2)