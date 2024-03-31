#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
from numba import njit, prange, set_num_threads

@njit(parallel=True)
def computeRegionFilter(QR, DLD, LVAR, sbnd):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       Q = np.empty((LVAR,2,1))
        
       for ii in prange(LVAR):
              # Compute the given filter over the region
              gval = np.nanmax(fltKrl[ii] @ QR[fltDex[ii],:])
              
              Q[ii,0,0] = min(DLD[2] * gval, sbnd)
              Q[ii,1,0] = min(DLD[3] * gval, sbnd)
                            
       return Q

def computeResidualViscCoeffs(RES, DLD, DT, bdex, sbnd, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Set DynSGS values
       CRES += computeRegionFilter(RES, DLD, LVAR, sbnd)
              
       return CRES