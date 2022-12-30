#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn
from numba import njit, prange
#import sparse_dot_mkl as spk

def computeResidualViscCoeffs(DIMS, state, RHS, RES, DLD, bdex, applyFilter):
       
       # Change floating point errors
       np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')
       
       # Compute flow speed components
       UD = np.abs(state[:,0])
       WD = np.abs(state[:,1])
       # Compute flow speed along terrain
       VD = np.sqrt(bn.ss(state[:,0:2], axis=1))
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)#; ARES[ARES < 1.0E-16] = 0.0
       RMAX = bn.nanmax(ARES,axis=0)
       RMAX[RMAX < 1.0E-16] = 1.0
       
       # Normalize each component residual and reduce to the measure on all variables
       QR = np.diag(np.reciprocal(RMAX))
       NARES = ARES @ QR
       CRES = bn.nanmax(NARES, axis=1); CRES[CRES <= 1.0E-16] = 0.0
       
       if applyFilter:
              nbrDex = DLD[-1] # List of lists of indices to regions
       
              # Apply the maximum filter over the precomputed regions
              CRESL = [bn.nanmax(CRES[reg]) for reg in nbrDex]
              CRES = np.array(CRESL)
       
       # Upper bound flow speed coefficients
       QMAX1 = 0.5 * DLD[0] * UD; QMAX1[bdex] = 0.5 * DLD[0][bdex] * VD[bdex]
       QMAX2 = 0.5 * DLD[1] * WD
       
       # Max norm of the limit coefficients
       QB1 = bn.nanmax(QMAX1)
       QB2 = bn.nanmax(QMAX2)
       
       # Residual based coefficients
       CRES1 = QB1 * CRES
       CRES2 = QB2 * CRES
       
       CRES1 = np.expand_dims(CRES1, axis=1)
       CRES2 = np.expand_dims(CRES2, axis=1)
       
       return (CRES1, CRES2)

@njit(parallel=True)
def computeRegionedArray(UD, WD, Q_RES, DLD, maxL, lstL, nbrDex):
       
       ii = 0
       LVAR = Q_RES.shape[0]
       Q_REG = np.zeros((LVAR,maxL,3))

       for ii in prange(LVAR):
              rdex = nbrDex[ii]
              Q_REG[ii,0:lstL[ii],0] = UD[rdex]
              Q_REG[ii,0:lstL[ii],1] = WD[rdex]
              Q_REG[ii,0:lstL[ii],2] = Q_RES[rdex]
              
       return Q_REG, LVAR

#@jit(nopython=True)
def computeResidualViscCoeffs2(DIMS, AV, RES, DLD, bdex, ldex, applyFilter, RLM, DCFC):
       
       # Change floating point errors
       #np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')
       
       # Get the region indices map
       nbrDex = DLD[-2] # List of lists of indices to regions
       maxL = DLD[-1][0]
       lstL = DLD[-1][1]
       
       # Compute flow speed along terrain
       UD = AV[:,0]
       WD = AV[:,1]
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       # Reduce across the variables using the 1-norm
       Q_RES = bn.nansum(ARES, axis=1)
       #Q_RES = np.nansum(ARES, axis=1)
       
       #%% Apply a region filter
       if applyFilter:
              Q_REG, LVAR = computeRegionedArray(UD, WD, Q_RES, DLD, maxL, lstL, nbrDex)
              RVAR = bn.nanmax(Q_REG, axis=1)
              UD = RVAR[:,0]
              WD = RVAR[:,1]
              Q_RES = RVAR[:,2]
       
       #%% LIMIT THE RESIDUAL COEFFICIENTS TO THE FLOW SPEED VALUES LOCALLY
       QC1 = np.stack((0.5 * DLD[0] * UD, DLD[2] * Q_RES), axis=-1)
       QC2 = np.stack((0.5 * DLD[1] * WD, DLD[3] * Q_RES), axis=-1)
       QC = np.vstack((QC1,QC2))
       QCM = bn.nanmin(QC, axis=1)
       
       CRES1 = np.expand_dims(QCM[0:Q_RES.shape[0]], axis=1)
       CRES2 = np.expand_dims(QCM[Q_RES.shape[0]:], axis=1)
       
       #%% SET INTERIOR CONSTANT DAMPING
       CRES1 += DCFC[1]
       CRES2 += DCFC[2]
       
       #%% SET DAMPING WITHIN THE ABSORPTION LAYERS
       CRES1[ldex,0] += DCFC[0] * RLM[0,ldex]
       CRES2[ldex,0] += DCFC[0] * RLM[0,ldex]
       
       return (CRES1, CRES2)