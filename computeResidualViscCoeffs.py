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
                     rsum = (np.exp(vals - rsmx)).sum()                     
                     gval = rsmx #+ np.log(rsum)
              else:
                     gval = vals.T @ fltKrl[ii]
              
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