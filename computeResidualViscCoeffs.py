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

def computeResidualViscCoeffs(DIMS, RES, QM, UD, WD, DLD, DLD2, filtType):
       #'''
       # Compute magnitude of flow speed
       vel = np.stack((UD, WD),axis=1)
       VFLW = np.linalg.norm(vel, axis=1)
       #'''
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Pixel size for image filter
       MFS = (6,6)
       GFS = (2,2)
       
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
                     elif filtType == 'gaussian':
                            ARES_XZ = ndimage.gaussian_filter(ARES_IM, sigma=GFS, mode='nearest')
                     elif filtType == 'none':
                            ARES_XZ = ARES_IM
                     else:
                            # Default to the mean filter
                            ARES_XZ = ndimage.uniform_filter(ARES_IM, size=MFS, mode='nearest')
                            
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
              QMAXF_FL = ndimage.maximum_filter(QMAXF, size=MFS, mode='nearest')
       elif filtType == 'median':
              QMAXF_FL = ndimage.median_filter(QMAXF, size=MFS, mode='nearest')
       elif filtType == 'mean':
              QMAXF_FL = ndimage.uniform_filter(QMAXF, size=MFS, mode='nearest')
       elif filtType == 'gaussian':
              QMAXF_FL = ndimage.gaussian_filter(QMAXF, sigma=GFS, mode='nearest')
       elif filtType == 'none':
              QMAXF_FL = QMAXF
       else:
              # Default to the mean filter
              QMAXF_FL = ndimage.uniform_filter(QMAXF, size=MFS, mode='nearest')
       
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

def computeResidualViscCoeffs3(DIMS, RES, QM, VFLW, DLD, DLD2):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Pixel size for image filter
       MFS = (3,3)
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       for vv in range(4):
              if QM[vv] > 0.0:
                     # Apply image filter to each residual to a pixel neigborhood
                     ARES_IM = np.reshape(ARES[:,vv], (NZ, NX), order='F')
                     ARES_XZ = ndimage.maximum_filter(ARES_IM, size=MFS, mode='mirror')
                     ARES[:,vv] = np.reshape(ARES_XZ, (OPS,), order='F')
                     
                     # Normalize the residuals
                     if vv < 4:
                            ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX = bn.nanmax(ARES, axis=1)
       #'''
       # Compute upper bound on coefficients (single bounding field)
       QMAXF = np.reshape(0.5 * mt.sqrt(DLD2) * VFLW, (NZ, NX), order='F')
       QMAXF = ndimage.maximum_filter(QMAXF, size=MFS, mode='mirror')
       QMAX = np.reshape(QMAXF, (OPS,), order='F')
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD[0]**2 * QRES_MAX, QMAX),axis=1)
       CRES1 = bn.nanmin(compare, axis=1)
       
       compare = np.stack((DLD[1]**2 * QRES_MAX, QMAX),axis=1)
       CRES2 = bn.nanmin(compare, axis=1)
       
       return (CRES1, CRES2)

def computeResidualViscCoeffs4(DIMS, RES, QM, VFLW, DLD, DLD2, NEX):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Pixel size for image filter
       if NEX == None:
              MFS = (4,4)
       else:
              MFS = (int(NEX/2) + 1, 4)
       
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
       # Compute upper bound on coefficients (single bounding field)
       QMAX = 0.5 * mt.sqrt(DLD2) * VFLW
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD[0]**2 * QRES_MAX, QMAX),axis=1)
       CRES1 = bn.nanmin(compare, axis=1)
       CRES1XZ = np.reshape(CRES1, (NZ, NX), order='F')
       CRES1XZ = ndimage.maximum_filter(CRES1XZ, size=MFS, mode='mirror')
       CRES1 = np.reshape(CRES1XZ, (OPS,), order='F')
       
       compare = np.stack((DLD[1]**2 * QRES_MAX, QMAX),axis=1)
       CRES2 = bn.nanmin(compare, axis=1)
       CRES2XZ = np.reshape(CRES2, (NZ, NX), order='F')
       CRES2XZ = ndimage.maximum_filter(CRES2XZ, size=MFS, mode='mirror')
       CRES2 = np.reshape(CRES2XZ, (OPS,), order='F')
       
       return (CRES1, CRES2)
