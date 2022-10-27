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

def computeResidualViscCoeffs(DIMS, RES, state, DLD, bdex, applyFilter):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       numVar = 1
       
       # Compute flow speed components
       UD = np.abs(state[:,0]); UD[UD < 1.0E-16] = 0.0
       WD = np.abs(state[:,1]); WD[WD < 1.0E-16] = 0.0
       # Compute flow speed along terrain
       VD = np.sqrt(bn.ss(state[bdex,0:2], axis=1))
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       RMAX = bn.nanmax(ARES,axis=0)
       RMAX[RMAX < 1.0E-16] = 1.0
       
       # Normalize each component residual and reduce to the max on all variables
       QR = np.diag(np.reciprocal(RMAX))
       CRES = bn.nanmax(ARES @ QR, axis=1)
       
       # Apply flow dependence at the boundary
       CRES[bdex] = 1.0 / bn.nanmax(VD) * VD      
       
       if applyFilter:
              nbrDex = DLD[-1] # List of lists of indices to regions
       
              # Apply the maximum filter over the precomputed regions
              CRESL = [bn.nanmax(CRES[reg]) for reg in nbrDex]
              CRES = np.array(CRESL)
       
       # Interior flow speed coefficients
       QMAX1 = 0.5 * DLD[0] * UD
       QMAX2 = 0.5 * DLD[1] * WD
       
       # Max norm of the limit coefficients
       QB1 = bn.nanmax(QMAX1)
       QB2 = bn.nanmax(QMAX2)
       
       # Residual based coefficients
       CRES1 = QB1 * CRES; CRES1[CRES1 < 1.0E-16] = 0.0
       CRES2 = QB2 * CRES; CRES2[CRES2 < 1.0E-16] = 0.0
       
       CRES1 = np.expand_dims(CRES1, axis=1)
       CRES2 = np.expand_dims(CRES2, axis=1)
       
       '''
       # Max norm of the variable coefficients
       QR1 = bn.nanmax(CRES1, axis=0)
       QR2 = bn.nanmax(CRES2, axis=0)
       
       for vv in range(4):
              CR1 = CRES1[:,vv]
              CR2 = CRES2[:,vv]
              
              CR1[CR1 < 1.0E-16] = 0.0
              CR2[CR2 < 1.0E-16] = 0.0
              
              CR1 *= QB1 / QR1[vv]
              CR2 *= QB2 / QR2[vv]
              CRES1[:,vv] = CR1
              CRES2[:,vv] = CR2
       '''       
       # Apply a simple post-filter
       '''
       if applyFilter:
              CRES1_XZ = np.reshape(CRES1, (NZ, NX, numVar), order='F')
              CRES2_XZ = np.reshape(CRES2, (NZ, NX, numVar), order='F')
              
              CRES1_XZ_FT = np.empty(CRES1_XZ.shape)
              CRES2_XZ_FT = np.empty(CRES2_XZ.shape)
              for vv in range(numVar):
                     CRES1_XZ_FT[:,:,vv] = ndimage.maximum_filter(CRES1_XZ[:,:,vv], size=4, mode='nearest')
                     CRES2_XZ_FT[:,:,vv] = ndimage.maximum_filter(CRES2_XZ[:,:,vv], size=4, mode='nearest')
              
              CRES1 = 1.0 * np.reshape(CRES1_XZ_FT, (OPS,numVar), order='F')
              CRES2 = 1.0 * np.reshape(CRES2_XZ_FT, (OPS,numVar), order='F')
       else:
              CRES1 = np.expand_dims(CRES1, axis=1)
              CRES2 = np.expand_dims(CRES2, axis=1)
       '''
       return (CRES1, CRES2)