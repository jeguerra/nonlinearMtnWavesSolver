#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

byLocalFilter = True
import math as mt
import numpy as np
import bottleneck as bn
from numba import njit, prange, set_num_threads

@njit(parallel=True)
def computeRegionFilter(Q, QR, DLD, LVAR):
       
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
        
       for ii in prange(LVAR):
              # Compute the given filter over the region
              gval = np.nanmax(fltKrl[ii] @ QR[fltDex[ii],:])
              #gval = np.nanmax(np.nanmax(QR[fltDex[ii],:]))
              
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

def computeResidualViscCoeffs(RES, BND, DLD, bdex, ldex, RLM, SMAX, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # DynSGS bounds by advective speed
       ubnd = 0.5 * DLD[0] * BND[0]
       wbnd = 0.5 * DLD[1] * BND[1]
       
       set_num_threads(10)
       if byLocalFilter:
           CRES = computeRegionFilter(CRES, RES, DLD, LVAR)
           
           CRES[:,0,0] = np.where(CRES[:,0,0] > ubnd, ubnd, CRES[:,0,0])
           CRES[:,1,0] = np.where(CRES[:,1,0] > wbnd, wbnd, CRES[:,1,0])
       else:
           Q_RES = bn.nanmax(RES, axis=1) 
           
           qr = DLD[2] * Q_RES
           CRES[:,0,0] = np.where(qr > ubnd, ubnd, qr)
           qr = DLD[3] * Q_RES
           CRES[:,1,0] = np.where(qr > wbnd, wbnd, qr)
           
           CRES = computeRegionFilter1(CRES, DLD, LVAR)

       # Augment damping to the sponge layers
       CRES[ldex,0,0] += ubnd * RLM[0,ldex]
       CRES[ldex,1,0] += wbnd * RLM[0,ldex]
              
       return CRES