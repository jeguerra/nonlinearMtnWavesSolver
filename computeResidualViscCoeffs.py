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
       
       # Compute upper bound on coefficients based on flow speed
       QMAX1 = 0.5 * DLD[0] * UD
       QMAX2 = 0.5 * DLD[1] * WD
       '''
       if applyFilter:
              QMAX1 = ndimage.maximum_filter(np.reshape(QMAX1, (NZ, NX), order='F'), size=3, mode='nearest')
              QMAX2 = ndimage.maximum_filter(np.reshape(QMAX2, (NZ, NX), order='F'), size=3, mode='nearest')
              QMAX1 = np.reshape(QMAX1, (OPS,), order='F')
              QMAX2 = np.reshape(QMAX2, (OPS,), order='F')
       '''
       # Compute field normalization
       QM = bn.nanmax(np.abs(qnorm), axis=0)
       QM = np.reciprocal(QM)
       QM[QM == np.inf] = 0.0
       QM = np.diag(QM)
       
       try:   
              # Compute absolute value of residuals
              ARES = np.abs(RES) @ QM
              '''
              # Apply image filtering to a 3 grid neighborhood
              if applyFilter:
                     QRES_PER = np.reshape(ARES, (NZ,NX,4), order='F')
                     QRES_FLT = np.empty(QRES_PER.shape)
                     for vv in range(4):
                            QRES_FLT[:,:,vv] = ndimage.maximum_filter(QRES_PER[:,:,vv], size=3, mode='nearest')
                     
                     ARES = np.reshape(QRES_FLT, (OPS,4), order='F')
              '''
              # Apply length scales to the coefficients
              CRES1 = DLD[2] * ARES
              CRES2 = DLD[3] * ARES

              # Apply limiting bounds
              QB1 = bn.nanmax(QMAX1)
              QB2 = bn.nanmax(QMAX2)
              
              for vv in range(4):
                     CR1 = CRES1[:,vv]
                     CR2 = CRES2[:,vv]
                     CR1[CR1 > QB1] = QB1
                     CR2[CR2 > QB2] = QB2
                     CRES1[:,vv] = CR1
                     CRES2[:,vv] = CR2

              # Set U and W damping along terrain to maximum flow dependent coefficients
              #CRES1[bdex,0] = QMAX1[bdex]
              #CRES2[bdex,1] = QMAX2[bdex]
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
                     CRES1_XZ_FT[:,:,vv] = ndimage.maximum_filter(CRES1_XZ[:,:,vv], size=3, mode='nearest')
                     CRES2_XZ_FT[:,:,vv] = ndimage.maximum_filter(CRES2_XZ[:,:,vv], size=3, mode='nearest')
              
              CRES1 = np.reshape(CRES1_XZ_FT, (OPS,4), order='F')
              CRES2 = np.reshape(CRES2_XZ_FT, (OPS,4), order='F')
       #'''
       return (CRES1, CRES2)