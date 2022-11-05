#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import math as mt
import numpy as np
import bottleneck as bn
from scipy import ndimage

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

def computeResidualViscCoeffs2(DIMS, state, RHS, RES, DLD, bdex, applyFilter):
       
       # Change floating point errors
       np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')
       
       QC1 = np.empty((DIMS[5],2))
       QC2 = np.empty((DIMS[5],2))
       
       #%% FLOW SPEED COEFFICIENTS
       
       # Compute flow speed components
       UD = np.abs(state[:,0])
       WD = np.abs(state[:,1])
       # Compute flow speed along terrain
       VD = np.sqrt(bn.ss(state[:,0:2], axis=1))
       
       # Upper bound flow speed coefficients
       QC1[:,0] = 0.5 * DLD[0] * UD; QC1[bdex,0] = 0.5 * DLD[0][bdex] * VD[bdex]
       QC2[:,0] = 0.5 * DLD[1] * WD; QC2[bdex,0] = 0.0
       
       # Apply a region filter
       if applyFilter:
              nbrDex = DLD[-1] # List of lists of indices to regions
       
              # Apply the maximum filter over the precomputed regions
              QVL = [bn.nanmax(QC1[reg,0]) for reg in nbrDex]
              QC1[:,0] = np.array(QVL)
              QVL = [bn.nanmax(QC2[reg,0]) for reg in nbrDex]
              QC2[:,0] = np.array(QVL)
       
       #%% RESIDUAL/RHS COEFFICIENTS
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       # Compute the residual normalization based on the state
       stateMean = DLD[-2] @ state
       RMAX = bn.nanmax(np.abs(state - stateMean), axis=0)
       RMAX[RMAX <= 1.0E-16] = 1.0
       
       # Normalize each component residual and reduce to the measure on all variables
       QR = np.diag(np.reciprocal(RMAX))
       NARES = ARES @ QR
       CRES = bn.nanmax(NARES, axis=1); CRES[CRES <= 1.0E-16] = 0.0
       
       # Apply a region filter
       if applyFilter:
              nbrDex = DLD[-1] # List of lists of indices to regions
       
              # Apply the maximum filter over the precomputed regions
              CRESL = [bn.nanmax(CRES[reg]) for reg in nbrDex]
              CRES = np.array(CRESL)
       
       # Compute the anisotropic coefficients
       QC1[:,1] = DLD[2] * CRES
       QC2[:,1] = DLD[3] * CRES
       
       #%% LIMIT THE RESIDUAL COEFFICIENTS TO THE FLOW SPEED VALUES LOCALLY
       
       CRES1 = np.expand_dims(bn.nanmin(QC1, axis=1), axis=1)
       CRES2 = np.expand_dims(bn.nanmin(QC2, axis=1), axis=1)
       
       return (CRES1, CRES2)