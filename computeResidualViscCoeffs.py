#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
from numba import njit, prange, set_num_threads

@njit(parallel=True)
def computeRegionFilter(Q, QR, DLD, LVAR):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
        
       for ii in prange(LVAR):
              # Compute the given filter over the region
              gval = np.nanmax(fltKrl[ii] @ QR[fltDex[ii],:])
              
              Q[ii,0,0] = DLD[2] * gval
              Q[ii,1,0] = DLD[3] * gval
                            
       return Q
   
@njit(parallel=True)
def computeRegionFilter1(Q, DLD, LVAR):
       
       fval = np.empty(Q.shape)
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       
       for ii in prange(LVAR):
              # Compute the given filter over the region
              fval[ii,0,0] = fltKrl[ii] @ Q[fltDex[ii],0,0]
              fval[ii,1,0] = fltKrl[ii] @ Q[fltDex[ii],1,0]
              
       return fval

@njit(parallel=True)
def computeRegionFilter2(Q, DLD, LVAR):
       
       fval = np.empty(Q.shape)
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       
       for ii in prange(LVAR):
              # Compute the given filter over the region
              fval[ii,:] = fltKrl[ii] @ Q[fltDex[ii],:]
              
       return fval

def computeResidualViscCoeffs(RES, BND, DLD, DT, bdex, RLM, VMAX, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # DynSGS bounds by advective speed
       sbnd = 0.5 * DT * VMAX**2
       
       set_num_threads(10)
       CRES = computeRegionFilter(CRES, RES, DLD, LVAR)
       CRES[:,:,0] = np.where(CRES[:,:,0] > sbnd, sbnd, CRES[:,:,0])

       # Augment damping to the sponge layers
       rlfc = sbnd * RLM
       CRES[:,:,0] += np.expand_dims(rlfc, axis=1)
              
       return CRES