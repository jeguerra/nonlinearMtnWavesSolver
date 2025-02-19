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
useLocalAverage = True

@njit(parallel=True)
def computeRegionFilter(residual, DLD, LVAR, sbnd):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       Q = np.empty((LVAR,2,1))
        
       for ii in prange(LVAR):
              
              # Get the target region values
              resv = residual[fltDex[ii]]
              rsmx = resv.max()
              
              # Compute the given filter over the region
              if useSmoothMaxFilter:                     
                     args = resv - rsmx
                     eargs = np.exp(args)
                     gval = rsmx + np.log(eargs.mean())
              else:
                     if useLocalAverage:
                            # Function average in window
                            gval = fltKrl[ii] @ resv
                     else:
                            # Function max in window
                            gval = rsmx
                            
              if gval < 1.0E-16:
                     gval = 0.0
              
              Q[ii,0,0] = min(DLD[2] * gval, sbnd)
              Q[ii,1,0] = min(DLD[3] * gval, sbnd)
                            
       return Q

def computeResidualViscCoeffs(RES, DLD, DT, bdex, sbnd):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Set DynSGS values averaged with previous 
       dcf = computeRegionFilter(RES, DLD, LVAR, sbnd)
              
       return dcf