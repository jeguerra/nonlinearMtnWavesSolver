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

def computeResidualViscCoeffs(DIMS, RES, QM, UD, WD, DX2, DZ2, DXZ, filtType):
       #'''
       # Compute magnitude of flow speed
       vel = np.stack((UD, WD),axis=1)
       VFLW = np.linalg.norm(vel, axis=1)
       # Compute a filter length...
       #
       #DXZ = DXD * DZD
       DL = np.sqrt(DXZ)
       #'''
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Pixel size for image filter
       MFS = (3,3)
       GFS = (1,1)
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       for vv in range(4):
              if QM[vv] > 0.0:
                     # Apply image filter to each residual to a pixel neigborhood
                     ARES_IM = np.reshape(ARES[:,vv], (NZ, NX), order='F')
                     
                     if filtType == 'maximum':
                            ARES_XZ = ndimage.maximum_filter(ARES_IM, size=MFS, mode='nearest')
                     elif filtType == 'median':
                            ARES_XZ = ndimage.median_filter(ARES_IM, size=MFS, mode='nearest')
                     elif filtType == 'mean':
                            ARES_XZ = ndimage.uniform_filter(ARES_IM, size=MFS, mode='nearest')
                     else:
                            ARES_XZ = ndimage.gaussian_filter(ARES_IM, sigma=GFS, mode='nearest')
                            
                     ARES[:,vv] = np.reshape(ARES_XZ, (OPS,), order='F')
                     # Normalize the residuals
                     ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX = bn.nanmax(ARES, axis=1)
       #'''
       # Compute upper bound on coefficients
       QMAXF = np.reshape(0.5 * DL * VFLW, (NZ, NX), order='F')
       
       if filtType == 'maximum':
              QMAXF_FL = ndimage.maximum_filter(QMAXF, size=MFS, mode='nearest')
       elif filtType == 'median':
              QMAXF_FL = ndimage.median_filter(QMAXF, size=MFS, mode='nearest')
       elif filtType == 'mean':
              QMAXF_FL = ndimage.uniform_filter(QMAXF, size=MFS, mode='nearest')
       else:
              QMAXF_FL = ndimage.gaussian_filter(QMAXF, sigma=GFS, mode='nearest')
       
       QMAX = np.reshape(QMAXF_FL, (OPS,), order='F')
       #'''
       
       # Limit DynSGS to upper bound
       compare = np.stack((1.0 * DX2 * QRES_MAX, QMAX),axis=1)
       CRES1 = bn.nanmin(compare, axis=1)
       compare = np.stack((1.0 * DZ2 * QRES_MAX, QMAX),axis=1)
       CRES2 = bn.nanmin(compare, axis=1)

       #return (np.expand_dims(CRES,1), np.expand_dims(QMAX,1))
       return (CRES1, CRES2)

def computeResidualViscCoeffs2(DIMS, RES, QM, UD, WD, DXD, DZD, DX2, DZ2):
       #'''
       # Compute magnitude of flow speed
       vel = np.stack((UD, WD),axis=1)
       VFLW = np.linalg.norm(vel, axis=1)
       # Compute a filter length...
       DXZ = DXD * DZD
       DL = np.sqrt(DXZ)
       #'''
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       for vv in range(4):
              if QM[vv] > 0.0:
                     # Normalize the residuals
                     ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX = bn.nanmax(ARES, axis=1)
       #'''
       # Compute upper bound on coefficients
       QMAX = 0.5 * DL * VFLW
       #'''
       
       # Limit DynSGS to upper bound
       compare = np.stack((1.0 * DX2 * QRES_MAX, QMAX),axis=1)
       CRES1 = bn.nanmin(compare, axis=1)
       compare = np.stack((1.0 * DZ2 * QRES_MAX, QMAX),axis=1)
       CRES2 = bn.nanmin(compare, axis=1)

       #return (np.expand_dims(CRES,1), np.expand_dims(QMAX,1))
       return (CRES1, CRES2)
