#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import math as mt
import numpy as np
import bottleneck as bn
import scipy.sparse as sps
from scipy import ndimage

def computeResidualViscCoeffs(DIMS, RES, QM, VFLW, DLD, DLD2, filtType, NE):
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       for vv in range(4):
              if QM[vv] > 0.0:
                     # Apply image filter to each residual to a pixel neigborhood
                     ARES_IM = np.reshape(ARES[:,vv], (NZ, NX), order='F')
                     
                     if filtType == 'maximum':
                            ARES_XZ = ndimage.maximum_filter(ARES_IM, size=NE, mode='nearest')
                     elif filtType == 'median':
                            ARES_XZ = ndimage.median_filter(ARES_IM, size=NE, mode='nearest')
                     elif filtType == 'mean':
                            ARES_XZ = ndimage.uniform_filter(ARES_IM, size=NE, mode='nearest')
                     elif filtType == 'gaussian':
                            ARES_XZ = ndimage.gaussian_filter(ARES_IM, sigma=NE, mode='nearest')
                     elif filtType == 'none':
                            ARES_XZ = ARES_IM
                     else:
                            # Default to the mean filter
                            ARES_XZ = ndimage.uniform_filter(ARES_IM, size=NE, mode='nearest')
                            
                     ARES[:,vv] = np.reshape(ARES_XZ, (OPS,), order='F')
                     
                     # Normalize the residuals
                     if vv < 2:
                            ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX = bn.nanmax(ARES, axis=1)
       #'''
       # Compute upper bound on coefficients (single bounding field)
       QMAXF = np.reshape(0.5 * mt.sqrt(DLD2) * VFLW, (NZ, NX), order='F')
       
       if filtType == 'maximum':
              QMAXF_FL = ndimage.maximum_filter(QMAXF, size=NE, mode='nearest')
       elif filtType == 'median':
              QMAXF_FL = ndimage.median_filter(QMAXF, size=NE, mode='nearest')
       elif filtType == 'mean':
              QMAXF_FL = ndimage.uniform_filter(QMAXF, size=NE, mode='nearest')
       elif filtType == 'gaussian':
              QMAXF_FL = ndimage.gaussian_filter(QMAXF, sigma=NE, mode='nearest')
       elif filtType == 'none':
              QMAXF_FL = QMAXF
       else:
              # Default to the mean filter
              QMAXF_FL = ndimage.uniform_filter(QMAXF, size=NE, mode='nearest')
       
       QMAX = np.reshape(QMAXF_FL, (OPS,), order='F')
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD2 * QRES_MAX, QMAX),axis=1)
       CRES = bn.nanmin(compare, axis=1)
       
       #compare = np.stack((DLD[1]**2 * QRES_MAX, QMAX1),axis=1)
       #CRES2 = bn.nanmin(compare, axis=1)

       return (CRES, CRES)

def computeResidualViscCoeffs2(DIMS, RES, QM, UD, WD, DLD, DLD2):
       #'''
       # Compute magnitude of flow speed
       #vel = np.stack((UD, WD),axis=1)
       #VFLW = np.linalg.norm(vel, axis=1)
       #'''
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       ARES1 = np.zeros(ARES.shape)
       ARES2 = np.zeros(ARES.shape)
       
       CF = np.ones((NZ, NX))
       
       for vv in range(4):
              if QM[vv] > 0.0:
                     # Apply image filter to each residual to a pixel neigborhood
                     ARES_IM = np.reshape(ARES[:,vv], (NZ, NX), order='F')
                     
                     xMaxes = sps.diags(bn.nanmax(ARES_IM, axis=1), offsets=0, format='csr')
                     ARES_X = xMaxes.dot(CF)
                     zMaxes = sps.diags(bn.nanmax(ARES_IM, axis=0), offsets=0, format='csr')
                     ARES_Z = (zMaxes.dot(CF.T)).T
                     
                     # Normalize the residuals
                     if vv < 4:
                            ARES_X *= (1.0 / QM[vv])
                            ARES_Z *= (1.0 / QM[vv])
                            
                     ARES1[:,vv] = np.reshape(ARES_X, (OPS,), order='F')
                     ARES2[:,vv] = np.reshape(ARES_Z, (OPS,), order='F')
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES1 = bn.nanmax(ARES1, axis=1)
       QRES2 = bn.nanmax(ARES2, axis=1)
       #'''
       # Compute upper bound on coefficients (single bounding field)
       QMAXF1 = np.reshape(0.5 * DLD[0] * UD, (NZ, NX), order='F')
       QMAXF2 = np.reshape(0.5 * DLD[1] * WD, (NZ, NX), order='F')
       
       xMaxes = sps.diags(bn.nanmax(QMAXF1, axis=1), offsets=0, format='csr')
       QMAXF_X = xMaxes.dot(CF)
       zMaxes = sps.diags(bn.nanmax(QMAXF2, axis=0), offsets=0, format='csr')
       QMAXF_Z = (zMaxes.dot(CF.T)).T
       
       QMAX1 = np.reshape(QMAXF_X, (OPS,), order='F')
       QMAX2 = np.reshape(QMAXF_Z, (OPS,), order='F')
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD[0]**2 * QRES1, QMAX1),axis=1)
       CRES1 = bn.nanmin(compare, axis=1)
       
       compare = np.stack((DLD[1]**2 * QRES2, QMAX2),axis=1)
       CRES2 = bn.nanmin(compare, axis=1)

       return (CRES1, CRES2)

def computeResidualViscCoeffs3(DIMS, RES, QM, VFLW, DLD, DLD2, NE):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       DL = mt.sqrt(DLD2)
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       for vv in range(4):
              if QM[vv] > 0.0:
                     # Normalize the residuals
                     if vv < 4:
                            ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX = bn.nanmax(ARES, axis=1)
       #'''
       # Compute upper bound on coefficients (single bounding fields
       QMAX = 0.5 * DL * VFLW
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD[0]**2 * QRES_MAX, QMAX),axis=1)
       CRES = bn.nanmin(compare, axis=1)
       CRES_XZ = np.reshape(CRES, (NZ, NX), order='F')
       CRES_XZ = ndimage.maximum_filter(CRES_XZ, size=NE, mode='mirror')
       CRES1 = np.reshape(CRES_XZ, (OPS,), order='F')
       
       compare = np.stack((DLD[1]**2 * QRES_MAX, QMAX),axis=1)
       CRES = bn.nanmin(compare, axis=1)
       CRES_XZ = np.reshape(CRES, (NZ, NX), order='F')
       CRES_XZ = ndimage.maximum_filter(CRES_XZ, size=NE, mode='mirror')
       CRES2 = np.reshape(CRES_XZ, (OPS,), order='F')
       
       return (np.expand_dims(CRES1, axis=1), np.expand_dims(CRES2, axis=1))

def computeResidualViscCoeffs4(DIMS, RES, QM, VFLW, DLD, DLD2):
       
       # Get the length scale
       DL = mt.sqrt(DLD2)
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       for vv in range(4):
              if QM[vv] > 0.0:
                     # Normalize the residuals
                     if vv < 4:
                            ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX = bn.nanmax(ARES, axis=1)
       #'''
       # Compute upper bound on coefficients (single bounding fields
       QMAX = 0.5 * DL * VFLW
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD[0]**2 * QRES_MAX, QMAX),axis=1)
       CRES1 = bn.nanmin(compare, axis=1)
       
       compare = np.stack((DLD[1]**2 * QRES_MAX, QMAX),axis=1)
       CRES2 = bn.nanmin(compare, axis=1)
       
       return (np.expand_dims(CRES1, axis=1), np.expand_dims(CRES2, axis=1))