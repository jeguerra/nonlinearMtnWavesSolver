#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
from numba import njit, prange

# Default is the local maximum for DynSGS coefficient
useSmoothMaxFilter = True
#useLocalAverage = True

@njit(parallel=True)
def computeRegionFilter(res_norm, residual, DLD, LVAR, sbnd):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       Q = np.empty((LVAR,2,1))
        
       for ii in prange(LVAR):
       
              vals = residual[fltDex[ii],:]
              resv = np.zeros((vals.shape[0]))
              for jj in prange(vals.shape[0]):
                     # Use the maximum from all residuals
                     resv[jj] = (vals[jj,:] * res_norm).max()
              
              # Compute the given filter over the region
              if useSmoothMaxFilter:                     
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
                     # Function average in window
                     gval = fltKrl[ii] @ resv
                     '''
                     if useLocalAverage:
                            # Function average in window
                            gval = fltKrl[ii] @ resv
                     else:
                            # Function max in window
                            gval = resv.max()
                     '''
              if gval < 1.0E-16:
                     gval = 0.0
              
              Q[ii,0,0] = min(0.5 * DLD[2] * gval, sbnd)
              Q[ii,1,0] = min(0.5 * DLD[3] * gval, sbnd)
                            
       return Q

def computeResidualViscCoeffs(res_norm, RES, DLD, DT, bdex, sbnd, dcf):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Set DynSGS values averaged with previous 
       dcf += computeRegionFilter(res_norm, RES, DLD, LVAR, sbnd)
              
       return dcf