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

def computeResidualViscCoeffs(DIMS, RES, QM, VFLW, DLD, DLD2, NE):
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
                     ARES_XZ = ndimage.maximum_filter(ARES_IM, size=NE, mode='nearest')
                            
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
       QMAXF_FL = ndimage.maximum_filter(QMAXF, size=NE, mode='nearest')
       
       QMAX = np.reshape(QMAXF_FL, (OPS,), order='F')
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD2 * QRES_MAX, QMAX),axis=1)
       CRES = bn.nanmin(compare, axis=1)

       return (np.expand_dims(CRES, axis=1), np.expand_dims(CRES, axis=1))

def computeResidualViscCoeffsRaw(DIMS, RES, qnorm, state, DLD, bdex, applyFilter):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Compute flow speed
       UD = np.abs(state[:,0])
       WD = np.abs(state[:,1])
       
       try:
              UD2 = np.power(UD,2.0)
       except FloatingPointError:
              UD2 = np.zeros(UD.shape)
              
       try:
              WD2 = np.power(WD,2.0)
       except FloatingPointError:
              WD2 = np.zeros(WD.shape)
       
       VL = np.sqrt(UD2 + WD2)
       
       # Compute field normalization
       QM = bn.nanmax(np.abs(qnorm), axis=0)
       
       try:   
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
              
              # Compute upper bound on coefficients based on flow speed
              QMAX1 = 0.5 * DLD[0] * UD
              QMAX2 = 0.5 * DLD[1] * WD
              
              # Apply image filtering to a 3 grid neighborhood
              if applyFilter:
                     QRES_MAX = ndimage.maximum_filter(np.reshape(QRES_MAX, (NZ, NX), order='F'), size=3, mode='constant')
                     QMAX1 = ndimage.maximum_filter(np.reshape(QMAX1, (NZ, NX), order='F'), size=3, mode='constant')
                     QMAX2 = ndimage.maximum_filter(np.reshape(QMAX2, (NZ, NX), order='F'), size=3, mode='constant')
                     
                     QRES_MAX = np.reshape(QRES_MAX, (OPS,), order='F')
                     QMAX1 = np.reshape(QMAX1, (OPS,), order='F')
                     QMAX2 = np.reshape(QMAX2, (OPS,), order='F')
              
              QB1 = bn.nanmax(QMAX1)
              QB2 = bn.nanmax(QMAX2)
              
              # Limit DynSGS to upper bound
              CRES1 = DLD[2] * QRES_MAX
              vdex = np.argwhere(CRES1 > QB1)
              CRES1[vdex] = QB1 #QMAX1[vdex]
              
              CRES2 = DLD[3] * QRES_MAX
              vdex = np.argwhere(CRES2 > QB2)
              CRES2[vdex] = QB2 #QMAX2[vdex]
              #'''
              # Set adaptive damping along the terrain
              QMAX = 0.5 * DLD[0] * DLD[4] * VL[bdex]
              QB = bn.nanmax(QMAX)
              CRES = DLD[2] * DLD[5] * QRES_MAX[bdex]
              vdex = np.argwhere(CRES > QB)
              CRES[vdex] = QB
              
              CRES1[bdex] = CRES
              CRES2[bdex] = CRES
              #'''
       except FloatingPointError:
              CRES1 = np.zeros((state.shape[0]))
              CRES2 = np.zeros((state.shape[0]))
              
       # Apply a simple post-filter
       '''
       if applyFilter:
              CRES1_XZ = np.reshape(CRES1, (NZ, NX), order='F')
              CRES1_XZ_FT = ndimage.maximum_filter(CRES1_XZ, size=3, mode='constant')
              CRES1 = np.reshape(CRES1_XZ_FT, (OPS,), order='F')
              
              CRES2_XZ = np.reshape(CRES2, (NZ, NX), order='F')
              CRES2_XZ_FT = ndimage.maximum_filter(CRES2_XZ, size=3, mode='constant')
              CRES2 = np.reshape(CRES2_XZ_FT, (OPS,), order='F')
       '''
       return (CRES1, CRES2)

def computeResidualViscCoeffsFiltered(DIMS, RES, fields, hydroState, DLD, NE):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Compute flow speed
       UD = fields[:,0] + hydroState[:,0]
       WD = fields[:,1]
       
       # Compute field normalization
       QM = bn.nanmax(np.abs(fields), axis=0)
       
       # Get the length scale
       DL = DLD[3]
       
       # Compute the flow magnitude
       vel = np.stack((UD, WD),axis=1)
       VFLW = np.linalg.norm(vel, axis=1)
       
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
       QRES_MAX_XZ = np.reshape(QRES_MAX, (NZ, NX), order='F')
       QRES_MAX_XZ_FT = ndimage.maximum_filter(QRES_MAX_XZ, size=NE, mode='nearest')
       QRES_MAX_FT = np.reshape(QRES_MAX_XZ_FT, (OPS,), order='F')
       
       # Compute upper bound on coefficients (single bounding fields
       QMAX = 0.5 * DL * VFLW
       QMAX_XZ = np.reshape(QMAX, (NZ, NX), order='F')
       QMAX_XZ_FT = ndimage.maximum_filter(QMAX_XZ, size=NE, mode='nearest')
       QMAX_FT = np.reshape(QMAX_XZ_FT, (OPS,), order='F')
       
       # Limit DynSGS to upper bound
       compare = np.stack((DLD[0]**2 * QRES_MAX_FT, QMAX_FT),axis=1)
       CRES1 = bn.nanmin(compare, axis=1)
       
       compare = np.stack((DLD[1]**2 * QRES_MAX_FT, QMAX_FT),axis=1)
       CRES2 = bn.nanmin(compare, axis=1)
       
       return (np.expand_dims(CRES1, axis=1), np.expand_dims(CRES2, axis=1))