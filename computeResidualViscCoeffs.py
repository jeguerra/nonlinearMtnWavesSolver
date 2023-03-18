#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
from numba import njit, prange

#@njit(parallel=True)
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
              
              # Compute the minimum bound on coefficient
              Q1[ii,0] = min(0.5 * DLD[0] * UDM, DLD[2] * QDM)
              Q2[ii,0] = min(0.5 * DLD[1] * WDM, DLD[3] * QDM)
              
       return Q1, Q2

def computeResidualViscCoeffs2(PHYS, AV, MAG, DLD, bdex, ldex, applyFilter, RLM, DCFC):
       
       # Get the region indices map
       nbrDex = DLD[-1] # List of lists of indices to regions
       
       # Compute flow speed along terrain
       UD = AV[:,0]# + DCFC[1]
       WD = AV[:,1]# + DCFC[1]
       
       # Compute absolute value of residuals
       AMAG = np.abs(MAG)
       LVAR = MAG.shape[0]
       
       # Diffusion proportional to the residual entropy
       Q_RES = PHYS[2] * AMAG[:,3] #bn.nansum(AMAG[:,2:], axis=1)
       
       #%% Filter to spatial regions and apply stability bounds
       if applyFilter:
              CRES1, CRES2 = computeRegionFilterBound(UD, WD, Q_RES, DLD, nbrDex, LVAR)
       else:
              CRES1 = DLD[2] * Q_RES
              CRES2 = DLD[3] * Q_RES
              
              QMAX = 0.5 * DLD[0] * UD
              mdex = np.nonzero(CRES1 > QMAX)
              CRES1[mdex] = QMAX[mdex]
              
              QMAX = 0.5 * DLD[1] * WD
              mdex = np.nonzero(CRES2 > QMAX)
              CRES2[mdex] = QMAX[mdex]
              
              CRES1 = np.expand_dims(CRES1, axis=1)
              CRES2 = np.expand_dims(CRES2, axis=1)
       
       #%% SET DAMPING WITHIN THE ABSORPTION LAYERS
       CRES1[ldex,0] += DCFC[0] * RLM[0,ldex]
       CRES2[ldex,0] += DCFC[0] * RLM[0,ldex]
       
       return (CRES1, CRES2)