#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import math as mt
import numpy as np
import bottleneck as bn

# This approach blends by maximum residuals on each variable
def computeResidualViscCoeffs(RES, QM, U, W, DX, DZ, DX2, DZ2, RLM, SV):
       
       ARES = np.abs(RES)
       
       # Normalize the residuals (U and W only!)
       for vv in range(2):
              if QM[vv] > 0.0:
                     ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals
       QRES_MAX = bn.nanmax(ARES, axis=1)
       
       # Compute the anisotropic coefficients
       QRESX = DX2 * QRES_MAX;
       QRESZ = DZ2 * QRES_MAX;
       
       # Compute upwind flow dependent coefficients
       QXMAX = (0.5 * DX) * U
       QZMAX = (0.5 * DZ) * W
       #'''
       compare = np.stack((QRESX, QXMAX),axis=1)
       QRESX_CF = bn.nanmin(compare, axis=1)
       compare = np.stack((QRESZ, QZMAX),axis=1)
       QRESZ_CF = bn.nanmin(compare, axis=1)
       #'''
       
       # Upwind diffusion in the sponge layers
       QRESX_CF += RLM.dot(QXMAX)
       QRESZ_CF += RLM.dot(QZMAX)
       
       # Scale with specific volume
       #QRESX_CF *= SV
       #QRESZ_CF *= SV
       
       return (np.expand_dims(QRESX_CF,1), np.expand_dims(QRESZ_CF,1))

# This approach keeps each corresponding residual on each variable
def computeResidualViscCoeffs2(RES, QM, U, W, DX, DZ, DX2, DZ2, RLM):
       
       ARES = np.abs(RES)
       
       # Normalize the residuals
       #'''
       for vv in range(4):
              if QM[vv] > 0.0:
                     ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
       
       # Compute the anisotropic coefficients
       QRESX = DX2 * ARES
       QRESZ = DZ2 * ARES
       
       # Compute upwind flow dependent coefficients
       QXMAX = (0.5 * DX) * U
       QZMAX = (0.5 * DZ) * W

       for vv in range(4):
              compare = np.stack((QRESX[:,vv], QXMAX),axis=1)
              QRESX[:,vv] = bn.nanmin(compare, axis=1)
              compare = np.stack((QRESZ[:,vv], QZMAX),axis=1)
              QRESZ[:,vv] = bn.nanmin(compare, axis=1)
      
       return (QRESX, QRESZ)

def computeFlowVelocityCoeffs(U, W, DX, DZ):
                     
       QRESX = np.zeros((len(U), 1))
       QRESZ = np.zeros((len(W), 1))
       
       # Compute the anisotropic coefficients
       QRESX[:,0] = (0.5 * DX) * U
       QRESZ[:,0] = (0.5 * DZ) * W
       
       return (QRESX, QRESZ)

def computeFlowAccelerationCoeffs(RES, DT, U, W, DX, DZ):
       
       ARES = np.abs(RES)
              
       QRESX = np.zeros((len(U), 4))
       QRESZ = np.zeros((len(W), 4))
       
       for vv in range(4):
              # Compute the anisotropic coefficients
              QRESX[:,vv] = (DX * DT) * ARES[0,vv]
              QRESZ[:,vv] = (DZ * DT) * ARES[1,vv]

       return (QRESX, QRESZ)