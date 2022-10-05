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

def computeResidualViscCoeffs(DIMS, RES, qnorm, state, DLD, bdex, applyFilter):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Compute flow speed
       UD = np.abs(state[:,0])
       WD = np.abs(state[:,1])
       #VD = np.sqrt(np.power(state[:,0],2.0) + np.power(state[:,1],2.0))
       
       # Compute upper bound on coefficients based on flow speed
       QMAX1 = 0.5 * DLD[0] * UD
       QMAX2 = 0.5 * DLD[1] * WD
       
       # Compute field normalization
       #QM = np.reciprocal(qnorm)
       #QM[QM == np.inf] = 0.0
       #QM = np.diag(QM.flatten())
       
       try:   
              # Compute absolute value of residuals
              ARES = np.abs(RES)# @ QM
              
              # Apply length scales to the coefficients
              CRES1 = DLD[2] * ARES
              CRES2 = DLD[3] * ARES
              
              # Max norm of the limit coefficients
              QB1 = bn.nanmax(QMAX1)
              QB2 = bn.nanmax(QMAX2)
              # Max norm of the variable coefficients
              QR1 = bn.nanmax(CRES1, axis=0)
              QR2 = bn.nanmax(CRES2, axis=0)
              
              for vv in range(4):
                     CR1 = CRES1[:,vv]
                     CR2 = CRES2[:,vv]
                     CR1 *= QB1 / QR1[vv]
                     CR2 *= QB2 / QR2[vv]
                     CRES1[:,vv] = CR1
                     CRES2[:,vv] = CR2
                     
                     # Apply flow speed proportional damping along terrain
                     #CRES1[bdex,vv] = 0.5 * DLD[0][bdex] * UD[bdex]
                     #CRES2[bdex,vv] = 0.5 * DLD[1] * WD[bdex]
              
       except FloatingPointError:
              CRES1 = np.zeros((ARES.shape))
              CRES2 = np.zeros((ARES.shape))
              
       # Apply a simple post-filter
       #'''
       if applyFilter:
              CRES1_XZ = np.reshape(CRES1, (NZ, NX, 4), order='F')
              CRES2_XZ = np.reshape(CRES2, (NZ, NX, 4), order='F')
              
              CRES1_XZ_FT = np.empty(CRES1_XZ.shape)
              CRES2_XZ_FT = np.empty(CRES2_XZ.shape)
              for vv in range(4):
                     CRES1_XZ_FT[:,:,vv] = ndimage.maximum_filter(CRES1_XZ[:,:,vv], size=4, mode='nearest')
                     CRES2_XZ_FT[:,:,vv] = ndimage.maximum_filter(CRES2_XZ[:,:,vv], size=4, mode='nearest')
              
              CRES1 = np.reshape(CRES1_XZ_FT, (OPS,4), order='F')
              CRES2 = np.reshape(CRES2_XZ_FT, (OPS,4), order='F')
       #'''
       return (CRES1, CRES2)