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

def computeRegionFilterBound_GPU(UD, WD, Q_RES, DLD, nbrDex, LVAR):
       
       # SEND THESE ARRAYS TO THE GPU
       coef_array1 = np.column_stack((DLD[2] * Q_RES,0.5 * DLD[0] * UD))
       coef_array2 = np.column_stack((DLD[3] * Q_RES,0.5 * DLD[1] * WD))
       
       Q1 = np.empty((LVAR,1))
       Q2 = np.empty((LVAR,1))
       
       for ii in np.arange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              
              Q = np.nanquantile(coef_array1[rdex,:],0.95,axis=0)
              Q1[ii,0] = Q.min()
              Q = np.nanquantile(coef_array2[rdex,:],0.95,axis=0)
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
              QDM = np.nanquantile(Q_RES[rdex],0.95)
              UDM = np.nanquantile(UD[rdex],0.95)
              WDM = np.nanquantile(WD[rdex],0.95)
              
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
              
              Q1[ii,0] = np.nanquantile(CRES1[rdex],0.95)
              Q2[ii,0] = np.nanquantile(CRES2[rdex],0.95)
              
       return Q1, Q2

@njit(parallel=True)
def computeRegionFilter2(CRES, nbrDex, LVAR, RDIM):

       Q = np.full((LVAR,RDIM,2),np.nan)
       
       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              Q[ii,:len(rdex),0] = CRES[rdex,0]
              Q[ii,:len(rdex),1] = CRES[rdex,1]
              
       return Q

def computeResidualViscCoeffs2(PHYS, AV, MAG, DLD, bdex, ldex, applyFilter, RLM, DCFC):
       
       # Get the region indices map
       nbrDex = DLD[-1] # List of lists of indices to regions
       
       # Compute absolute value of residuals
       AMAG = np.abs(MAG[:,3])
       LVAR = MAG.shape[0]
       RDIM = DLD[-2]
       
       # Diffusion proportional to the residual entropy
       Q_RES = PHYS[2] * AMAG
       
       #%% Filter to spatial regions and apply stability bounds
       if applyFilter:
              CRES1, CRES2 = computeRegionFilterBound_CPU(AV[:,0], AV[:,1], Q_RES, DLD, nbrDex, LVAR)
       else:
              CRES = np.empty((LVAR,2,2))
              CD = 0.5
              CRES[:,0,0] = DLD[2] * Q_RES
              CRES[:,1,0] = CD * DLD[0] * AV[:,0]
              CRES[:,0,1] = DLD[3] * Q_RES
              CRES[:,1,1] = CD * DLD[1] * AV[:,1]
              CRES_gpu = cp.asarray(CRES)
              
              CRES = cp.nanmin(CRES_gpu,axis=1).get()
              CRES = computeRegionFilter2(CRES, nbrDex, LVAR, RDIM)
              
              CRES_gpu = cp.asarray(CRES)
              CRES = cp.nanmax(CRES_gpu, axis=1, keepdims=False)
              #CRES = np.nanmean(CRES_gpu, axis=1, keepdims=False)
              #CRES = cp.nanmedian(CRES_gpu, axis=1, keepdims=False)
       
       # Gather back to the CPU
       CRES1 = cp.expand_dims(CRES[:,0], axis=1).get()
       CRES2 = cp.expand_dims(CRES[:,1], axis=1).get()
       
       # Augment damping to the sponge layers
       CRES1[ldex,0] += DCFC * RLM[0,ldex]
       CRES2[ldex,0] += DCFC * RLM[0,ldex]
       
       CRES = None
       CRES_gpu = None
       
       return (CRES1,CRES2)