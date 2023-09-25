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
def computeRegionFilter1(Q, QR, DLD, nbrDex, LVAR):
       
       for ii in prange(LVAR):
              # Compute the given filter over the region
              gval = np.nanmax(QR[nbrDex[ii]])
              
              Q[ii,0,0] = DLD[2] * gval
              Q[ii,1,0] = DLD[3] * gval
              
       return Q

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
       
       #CR = CRES[:,0]
       CRES[CRES > DCFC] = DCFC
       #CR = CRES[:,1]
       #CRES[CR > DCFC,1] = DCFC

       # Augment damping to the sponge layers
       RLD = DCFC * RLM[0,ldex]
       CRES[ldex,0,0] += RLD
       CRES[ldex,1,0] += RLD
              
       return CRES