#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn
from numba import njit, prange, set_num_threads

@njit(parallel=True)
def computeRegionFilter(Q, QR, QB, DLD, LVAR, sval):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
        
       for ii in prange(LVAR):
              # Compute the given filter over the region
              gval = np.nanmax(fltKrl[ii] @ QR[fltDex[ii],:])
              
              #gval = np.nanmedian(QR[fltDex[ii]])
              #uval = np.nanmax(QB[fltDex[ii],0])
              #wval = np.nanmax(QB[fltDex[ii],1])
              
              #gval = fltKrl[ii] @ rval #QR[fltDex[ii]]
              uval = fltKrl[ii] @ QB[fltDex[ii],0]
              wval = fltKrl[ii] @ QB[fltDex[ii],1]
              
              Q[ii,0,0] = min(DLD[2] * gval, 0.5 * DLD[0] * uval)
              Q[ii,1,0] = min(DLD[3] * gval, 0.5 * DLD[1] * wval)
              
       return Q

def computeResidualViscCoeffs(PHYS, RES, Q_BND, NOR, DLD, bdex, ldex, RLM, SMAX, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Diffusion proportional to the residual entropy
       Q_NOR = np.where(NOR > 0.0, NOR, 1.0)
       N_RES = np.abs(RES) / Q_NOR
       
       #input(bn.nanmax(N_RES, axis=0))
       
       # Compute filtering convolution
       set_num_threads(8)
       CRES = computeRegionFilter(CRES, N_RES, Q_BND, DLD, LVAR, SMAX)

       # Augment damping to the sponge layers
       CRES[ldex,0,0] += 0.5 * DLD[4] * SMAX * RLM[0,ldex]
       CRES[ldex,1,0] += 0.5 * DLD[4] * SMAX * RLM[0,ldex]
              
       return CRES, N_RES