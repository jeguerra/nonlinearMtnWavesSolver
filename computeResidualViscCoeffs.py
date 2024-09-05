#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
from numba import njit, prange

useMaxFilter = True

@njit(parallel=True)
def computeRegionFilter(res_norm, residual, DLD, LVAR, sbnd):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       Q = np.empty((LVAR,2,1))
        
       for ii in prange(LVAR):
              '''
              # Fetch the fields in this region
              vals = state[fltDex[ii],:]
              
              # Compute the local normalization
              vals_avg = fltKrl[ii] @ vals
              res_norm = fltKrl[ii] @ np.abs(vals - vals_avg)
                            
              # Reduce across residual variables
              res_norm = res_norm[res_norm > NORM_THRES]
              vals = residual[fltDex[ii],:]
              vals = vals[:,res_norm > NORM_THRES]
              '''
              vals = residual[fltDex[ii],:]
              resv = np.zeros((vals.shape[0]))
              for jj in prange(vals.shape[0]):
                     resv[jj] = (vals[jj,:] * res_norm).max()
              
              # Compute the given filter over the region
              if useMaxFilter:                     
                     rsmx = resv.max()
                     
                     rsum = 0.0
                     nv = 0
                     for val in resv:
                            arg = val - rsmx
                            if arg < 0.0:
                                   rsum += np.exp(arg)
                                   nv += 1

                     if nv > 0:
                            gval = rsmx + np.log(rsum / nv)
                     else:
                            gval = rsmx
              else:
                     gval = fltKrl[ii] @ resv
                     
              if gval < 1.0E-16:
                     gval = 0.0
              
              Q[ii,0,0] = min(0.5 * DLD[2] * gval, sbnd)
              Q[ii,1,0] = min(0.5 * DLD[3] * gval, sbnd)
                            
       return Q

def computeResidualViscCoeffs(res_norm, RES, DLD, DT, bdex, sbnd, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Set DynSGS values
       CRES += computeRegionFilter(res_norm, RES, DLD, LVAR, sbnd)
              
       return CRES