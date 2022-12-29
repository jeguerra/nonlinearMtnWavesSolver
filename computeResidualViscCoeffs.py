#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import math as mt
import numpy as np
import bottleneck as bn
from numba import jit
import sparse_dot_mkl as spk

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

@jit(nopython=True)
def computeResidualViscCoeffs2(DIMS, state, RES, DLD, bdex, ldex, applyFilter, RLM, DCFC):
       
       # Change floating point errors
       #np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')
       
       # Get the region indices map
       maxL = DLD[-1][0]
       lstL = DLD[-1][1]
       
       # Compute flow speed along terrain
       AV = np.abs(state[:,0:2])
       UD = AV[:,0]
       WD = AV[:,1]
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       # Reduce across the variables using the 1-norm
       #Q_RES = bn.nansum(ARES, axis=1)
       Q_RES = np.sum(ARES, axis=1)
       
       #%% Apply a region filter
       if applyFilter:
              nbrDex = DLD[-2] # List of lists of indices to regions
              
              ii = 0
              LVAR = UD.shape[0] + WD.shape[0] + Q_RES.shape[0]
              Q_REG = np.zeros((LVAR,maxL))
       
              for reg in nbrDex:
                     Q_REG[ii,0:lstL[ii]] = UD[reg]
                     Q_REG[ii + UD.shape[0],0:lstL[ii]] = WD[reg]
                     Q_REG[ii + UD.shape[0] + WD.shape[0],0:lstL[ii]] = Q_RES[reg]
                     ii += 1
                     
              #RVAR = bn.nanmax(Q_REG, axis=1)
              RVAR = np.max(Q_REG.T)
              UD = RVAR[0:UD.shape[0]]
              WD = RVAR[UD.shape[0]:UD.shape[0] + WD.shape[0]]
              Q_RES = RVAR[UD.shape[0] + WD.shape[0]:LVAR]
       
       #%% LIMIT THE RESIDUAL COEFFICIENTS TO THE FLOW SPEED VALUES LOCALLY
       QC1 = np.stack((0.5 * DLD[0] * UD, DLD[2] * Q_RES), axis=-1)
       QC2 = np.stack((0.5 * DLD[1] * WD, DLD[3] * Q_RES), axis=-1)
       CRES1 = np.expand_dims(bn.nanmin(QC1, axis=1), axis=1)
       CRES2 = np.expand_dims(bn.nanmin(QC2, axis=1), axis=1)
       
       #%% SET DAMPING ALONG THE TERRAIN SURFACE
       #CRES1[bdex,0] = 0.5 * DLD[0] * UD[bdex]
       #CRES2[bdex,0] = 0.5 * DLD[1] * WD[bdex]
       
       #%% SET DAMPING WITHIN THE ABSORPTION LAYERS
       CRES1[ldex,0] += DCFC * RLM[0,ldex]
       CRES2[ldex,0] += DCFC * RLM[0,ldex]
       
       return (CRES1, CRES2)