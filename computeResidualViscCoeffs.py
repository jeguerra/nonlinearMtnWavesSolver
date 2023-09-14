#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn
from numba import njit, prange, set_num_threads, get_num_threads

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
def computeRegionFilter1(Q, QR, DLD, nbrDex, LVAR):
       
       for ii in prange(LVAR):
              # Compute the given filter over the region
              gval = np.nanmax(QR[nbrDex[ii]])
              
              Q[ii,0] = DLD[2] * gval
              Q[ii,1] = DLD[3] * gval
              
       return Q

@njit(parallel=True)
def computeRegionFilter2(Q2FILT, nbrDex, LVAR, RDIM):

       QFILT = np.full((LVAR,2,RDIM),np.nan)
       
       for ii in prange(LVAR):
              # Get the region index
              rdex = nbrDex[ii]
              QFILT[ii,0,:len(rdex)] = Q2FILT[rdex,0]
              QFILT[ii,1,:len(rdex)] = Q2FILT[rdex,1]
              
       return QFILT

def computeResidualViscCoeffs2(PHYS, MAG, DLD, bdex, ldex, RLM, DCFC, CRES):
       
       # Get the region indices map
       nbrDex = DLD[-1] # List of lists of indices to regions
       
       # Compute absolute value of residuals
       AMAG = np.abs(MAG[:,3])
       LVAR = MAG.shape[0]
       
       # Diffusion proportional to the residual entropy
       Q_RES = PHYS[2] * AMAG
       
       set_num_threads(8)
       CRES = computeRegionFilter1(CRES, Q_RES, DLD, nbrDex, LVAR)
       
       CR = CRES[:,0]
       CRES[CR > DCFC,0] = DCFC
       CR = CRES[:,1]
       CRES[CR > DCFC,1] = DCFC

       # Augment damping to the sponge layers
       CRES[ldex,:] += np.expand_dims(DCFC * RLM[0,ldex], axis=1)
       #CRES[ldex,1] += DCFC * RLM[0,ldex]
       
       # Give the correct dimensions for operations
       CRES1 = np.expand_dims(CRES[:,0], axis=1)
       CRES2 = np.expand_dims(CRES[:,1], axis=1)
              
       return (CRES1, CRES2)