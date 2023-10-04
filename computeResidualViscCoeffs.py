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
def computeRegionFilter1(Q, QR, QB, DLD, LVAR, sval):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
        
       for ii in prange(LVAR):
              # Compute the given filter over the region
              #gval = np.nanmedian(QR[fltDex[ii]])
              uval = np.nanmedian(QB[fltDex[ii],0])
              wval = np.nanmedian(QB[fltDex[ii],1])
              
              gval = fltKrl[ii] @ QR[fltDex[ii]]
              #uval = fltKrl[ii] @ QB[fltDex[ii],0]
              #wval = fltKrl[ii] @ QB[fltDex[ii],1]
              
              Q[ii,0,0] = min(DLD[2] * gval, 0.5 * DLD[0] * uval)
              Q[ii,1,0] = min(DLD[3] * gval, 0.5 * DLD[1] * wval)
              
       return Q

def computeResidualViscCoeffs2(PHYS, RES, Q_BND, NOR, DLD, bdex, ldex, RLM, SMAX, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Diffusion proportional to the residual entropy
       Q_NOR = np.ones((1,4))
       Q_NOR = np.where(NOR > 0.0, NOR, Q_NOR)
       Q_RES = bn.nanmax(np.abs(RES) / Q_NOR, axis=1)
       
       # Compute filtering convolution
       set_num_threads(8)
       CRES = computeRegionFilter1(CRES, Q_RES, Q_BND, DLD, LVAR, SMAX)

       # Augment damping to the sponge layers
       CRES[ldex,0,0] += 0.5 * DLD[4] * SMAX * RLM[0,ldex]
       CRES[ldex,1,0] += 0.5 * DLD[4] * SMAX * RLM[0,ldex]
              
       return CRES