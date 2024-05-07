#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn
from numba import njit, prange

useMaxFilter = True

@njit(parallel=True)
def computeRegionFilter(QR, DLD, LVAR, sbnd):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       Q = np.empty((LVAR,2,1))
        
       for ii in prange(LVAR):
              # Compute the given filter over the region
              vals = QR[fltDex[ii]]
              if useMaxFilter:                     
                     rsmx = vals.max()
                     
                     rsum = 0.0
                     nv = 0
                     for val in vals:
                            arg = val - rsmx
                            if arg < 0.0:
                                   rsum += np.exp(arg)
                                   nv += 1

                     if nv > 0:
                            gval = rsmx + np.log(rsum / nv)
                     else:
                            gval = rsmx
                     #print(rsmx, np.log(rsum / nv), gval)
                     #input(vals - rsmx)
              else:
                     gval = vals.T @ fltKrl[ii]
                     
              if gval < 1.0E-16:
                     gval = 0.0
              
              Q[ii,0,0] = min(DLD[2] * gval, sbnd)
              Q[ii,1,0] = min(DLD[3] * gval, sbnd)
                            
       return Q

def computeResidualViscCoeffs(RES, DLD, DT, bdex, sbnd, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Set DynSGS values
       RES = bn.nanmax(RES,axis=1)
       CRES += computeRegionFilter(RES, DLD, LVAR, sbnd)
              
       return CRES