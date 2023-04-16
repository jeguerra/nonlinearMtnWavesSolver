#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn
from numba import njit, prange

def computeRegionFilterBound_GPU(UD, WD, Q_RES, DLD, nbrDex, LVAR):
       
       # SEND THESE ARRAYS TO THE GPU
       coef_array1 = np.column_stack((DLD[2] * Q_RES,0.5 * DLD[0] * UD))
       coef_array2 = np.column_stack((DLD[3] * Q_RES,0.5 * DLD[1] * WD))
       
       Q1 = np.empty((LVAR,1))
       Q2 = np.empty((LVAR,1))
       
       for ii in np.arange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              
              Q = np.nanquantile(coef_array1[rdex,:],0.9,axis=0)
              Q1[ii,0] = Q.min()
              Q = np.nanquantile(coef_array2[rdex,:],0.9,axis=0)
              Q2[ii,0] = Q.min()
              
       # BRING RETURN ARRAYS FROM THE GPU
       return Q1, Q2

@njit(parallel=True)
def computeRegionFilterBound_CPU(UD, WD, Q_RES, DLD, nbrDex, LVAR):
              
       Q1 = np.empty((LVAR,1))
       Q2 = np.empty((LVAR,1))
       #SD = np.sqrt(np.power(UD,2) + np.power(WD,2))

       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              
              # Compute the local L-inf norm in a region
              '''
              QDM = np.nanquantile(Q_RES[rdex],0.9)
              UDM = np.nanquantile(UD[rdex],0.9)
              WDM = np.nanquantile(WD[rdex],0.9)
              '''
              QDM = np.nanmax(Q_RES[rdex])
              UDM = np.nanmax(UD[rdex])
              WDM = np.nanmax(WD[rdex])
              
              # Compute the minimum bound on coefficient
              Q1[ii,0] = min(0.5 * DLD[0] * UDM, DLD[2] * QDM)
              Q2[ii,0] = min(0.5 * DLD[1] * WDM, DLD[3] * QDM)
              
       return Q1, Q2

@njit(parallel=True)
def computeRegionFilter(CRES1, CRES2, nbrDex, LVAR):

       Q1 = np.empty((LVAR,1))
       Q2 = np.empty((LVAR,1))       

       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              
              Q1[ii,0] = np.nanquantile(CRES1[rdex],0.9)
              Q2[ii,0] = np.nanquantile(CRES2[rdex],0.9)
              
       return Q1, Q2

def computeResidualViscCoeffs2(PHYS, AV, MAG, DLD, bdex, ldex, applyFilter, RLM, DCFC):
       
       # Get the region indices map
       nbrDex = DLD[-1] # List of lists of indices to regions
       
       # Compute absolute value of residuals
       AMAG = np.abs(MAG[:,3])
       LVAR = MAG.shape[0]
       
       # Diffusion proportional to the residual entropy
       Q_RES = PHYS[2] * AMAG
       
       #%% Filter to spatial regions and apply stability bounds
       if applyFilter:
              CRES1, CRES2 = computeRegionFilterBound_GPU(AV[:,0], AV[:,1], Q_RES, DLD, nbrDex, LVAR)
       else:
              CRES1 = np.column_stack((DLD[2] * Q_RES, 0.5 * DLD[0] * AV[:,0]))
              CRES2 = np.column_stack((DLD[3] * Q_RES, 0.5 * DLD[1] * AV[:,1]))
              
              CRES1 = bn.nanmin(CRES1,axis=1)
              CRES2 = bn.nanmin(CRES2,axis=1)
              
              CRES1, CRES2 = computeRegionFilter(CRES1, CRES2, nbrDex, LVAR)
       
       #%% SET DAMPING WITHIN THE ABSORPTION LAYERS
       CRES1[ldex,0] += DCFC[0] * RLM[0,ldex]
       CRES2[ldex,0] += DCFC[0] * RLM[0,ldex]
       
       return (CRES1, CRES2)