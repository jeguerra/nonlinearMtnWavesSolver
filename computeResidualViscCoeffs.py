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
              gval = 0.1 * np.nanmax(fltKrl[ii] @ QR[fltDex[ii],:])
              #gval = np.nanmax(np.nanmax(QR[fltDex[ii],:]))
              
              Q[ii,0,0] = DLD[2] * gval
              Q[ii,1,0] = DLD[3] * gval
              
       return Q
   
@njit(parallel=True)
def computeRegionFilterOne(Q, DLD, LVAR):
       
       fval = np.empty(Q.shape)
       fltDex = DLD[-2]
       fltKrl = DLD[-1]
       
       for ii in prange(LVAR):
              # Compute the given filter over the region
              fval[ii,0,0] = fltKrl[ii] @ Q[fltDex[ii],0,0]
              fval[ii,1,0] = fltKrl[ii] @ Q[fltDex[ii],1,0]
              
       return fval

def computeResidualViscCoeffs(PHYS, RES, Q_BND, NOR, DLD, bdex, ldex, RLM, SMAX, CRES):
       
       # Compute absolute value of residuals
       LVAR = RES.shape[0]
       
       # Diffusion proportional to the residual entropy
       Q_NOR = np.where(NOR > 0.0, NOR, 1.0)
       N_RES = np.abs(RES) / Q_NOR
       
       set_num_threads(8)
       if byLocalFilter:
           CRES = computeRegionFilter(CRES, N_RES, DLD, LVAR)
           '''
           bval = mt.sqrt(NOR[0]**2 + NOR[1]**2)
           qb = 0.5 * DLD[0] * bval
           CRES[:,0,0] = np.where(CRES[:,0,0] > qb, qb, CRES[:,0,0])
           qb = 0.5 * DLD[1] * bval
           CRES[:,1,0] = np.where(CRES[:,1,0] > qb, qb, CRES[:,1,0])
           '''
       else:
           Q_RES = bn.nanmax(N_RES, axis=1) 
           
           qr = DLD[2] * Q_RES
           qb = 0.5 * DLD[0] * Q_BND[:,0]
           CRES[:,0,0] = np.where(qr > qb, qb, qr)
           qr = DLD[3] * Q_RES
           qb = 0.5 * DLD[1] * Q_BND[:,1]
           CRES[:,1,0] = np.where(qr > qb, qb, qr)
           
           CRES = computeRegionFilterOne(CRES, DLD, LVAR)

       # Augment damping to the sponge layers
       CRES[ldex,0,0] += 0.5 * DLD[4] * SMAX * RLM[0,ldex]
       CRES[ldex,1,0] += 0.5 * DLD[4] * SMAX * RLM[0,ldex]
              
       return CRES, Q_NOR