#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn
from numba import njit, prange
import cupy as cp

@njit(parallel=True)
def computeRegionFilterBound1(UD, WD, Q_RES, DLD, nbrDex, LVAR):
              
       Q1 = np.empty((LVAR,1))
       Q2 = np.empty((LVAR,1))
       #SD = np.sqrt(np.power(UD,2) + np.power(WD,2))

       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              
              # Compute the local L-inf norm in a region
              QDM = np.nanquantile(Q_RES[rdex],0.95)
              UDM = np.nanquantile(UD[rdex],0.95)
              WDM = np.nanquantile(WD[rdex],0.95)
              
              # Compute the minimum bound on coefficient
              Q1[ii,0] = min(0.5 * DLD[0] * UDM, DLD[2] * QDM)
              Q2[ii,0] = min(0.5 * DLD[1] * WDM, DLD[3] * QDM)
              
       return Q1, Q2

@njit(parallel=True)
def computeRegionFilter1(Q, CR00, CR10, CR01, CR11, nbrDex, LVAR):
       
       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              Q[ii,0,0,:len(rdex)] = CR00[rdex]
              Q[ii,1,0,:len(rdex)] = CR01[rdex]
              Q[ii,0,1,:len(rdex)] = CR10[rdex]
              Q[ii,1,1,:len(rdex)] = CR11[rdex]
              
       return Q

@njit(parallel=True)
def computeRegionFilter2(Q2FILT, nbrDex, LVAR, RDIM):

       QFILT = np.full((LVAR,RDIM,2),np.nan)
       
       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              QFILT[ii,:len(rdex),0] = Q2FILT[rdex,0]
              QFILT[ii,:len(rdex),1] = Q2FILT[rdex,1]
              
       return QFILT

def computeResidualViscCoeffs2(PHYS, AV, MAG, DLD, bdex, ldex, RLM, DCFC, CRES):
       
       # Get the region indices map
       nbrDex = DLD[-1] # List of lists of indices to regions
       
       # Compute absolute value of residuals
       AMAG = np.abs(MAG[:,3])
       LVAR = MAG.shape[0]
       RDIM = DLD[-2]
       
       # Diffusion proportional to the residual entropy
       Pr = 0.71
       Q_RES = PHYS[2] * AMAG
       
       #%% Compare residual coefficients to upwind
       CD = 0.5
       #CRES = np.full((LVAR,2,2,RDIM), np.nan)
       CRES00 = DLD[2] * Q_RES
       CRES01 = CD * DLD[0] * AV[:,0]
       CRES10 = DLD[3] * Q_RES
       CRES11 = CD * DLD[1] * AV[:,1]
       
       CRES = computeRegionFilter1(CRES, CRES00, CRES10, CRES01, CRES11, nbrDex, LVAR)
       
       # Compute regions THEN compare coefficients
       CRES = bn.nanmax(CRES, axis=3)
       CRES = bn.nanmin(CRES, axis=1)
       # Compare coefficients THEN compute regions
       #CRES = bn.nanmin(CRES, axis=1)
       #CRES = bn.nanmax(CRES, axis=2)
       
       # Give the correct dimensions for operations
       CRES1 = Pr * np.expand_dims(CRES[:,0], axis=1)
       CRES2 = Pr * np.expand_dims(CRES[:,1], axis=1)
       
       # Augment damping to the sponge layers
       CRES1[ldex,0] += DCFC * RLM[0,ldex]
       CRES2[ldex,0] += DCFC * RLM[0,ldex]
              
       return (CRES1,CRES2)