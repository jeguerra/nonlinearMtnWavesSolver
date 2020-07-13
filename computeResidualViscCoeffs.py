#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import bottleneck as bn

# This approach blends by maximum residuals on each variable
def computeResidualViscCoeffs(RES, QM, U, W, DX, DZ, VSND, RLM):
       
       ARES = np.abs(RES)
       
       # Normalize the residuals
       #'''
       for vv in range(4):
              # Prandtl number scaling to theta
              if vv == 3:
                     scale = 0.71 / 0.4
              else:
                     scale = 1.0
                     
              if QM[vv] > 0.0:
                     ARES[:,vv] *= (scale / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals
       QRES_MAX = bn.nanmax(ARES, axis=1)
       
       # Compute the anisotropic coefficients
       QRESX = DX**2 * QRES_MAX;
       QRESZ = DZ**2 * QRES_MAX;
       
       # Compute upwind flow dependent coefficients
       XMAX = (0.5 * DX) * U#(U + VSND)
       ZMAX = (0.5 * DZ) * W#(W + VSND)
       #'''
       # Continuous damping in the sponge layers
       compare = np.stack((QRESX, XMAX),axis=1)
       QRESX_CF = RLM.dot(XMAX) + bn.nanmin(compare, axis=1)
       compare = np.stack((QRESZ, ZMAX),axis=1)
       QRESZ_CF = RLM.dot(ZMAX) + bn.nanmin(compare, axis=1)
       '''
       # Continuous upwind PLUS residual
       compare = np.stack((QRESX, XMAX),axis=1)
       QRESX_CF = 0.5 * (XMAX + bn.nanmin(compare, axis=1))
       compare = np.stack((QRESZ, ZMAX),axis=1)
       QRESZ_CF = 0.5 * (ZMAX + bn.nanmin(compare, axis=1))
       '''
       return (np.expand_dims(QRESX_CF,1), np.expand_dims(QRESZ_CF,1))


# This approach keeps each corresponding residual on each variable
def computeResidualViscCoeffs2(RES, QM, U, W, DX, DZ, VSND):
       
       ARES = np.abs(RES)
       
       # Normalize the residuals
       #'''
       for vv in range(4):
              # Prandtl number scaling to theta
              if vv == 3:
                     scale = 0.71 / 0.4
              else:
                     scale = 1.0
                     
              if QM[vv] > 0.0:
                     ARES[:,vv] *= (scale / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
       
       # Compute the anisotropic coefficients
       QRESX = DX**2 * ARES;
       QRESZ = DZ**2 * ARES;

       XMAX = (0.5 * DX) * U#(U + VSND)
       ZMAX = (0.5 * DZ) * W#(W + VSND)

       for vv in range(4):
              compare = np.stack((QRESX[:,vv], XMAX),axis=1)
              QRESX[:,vv] = bn.nanmin(compare, axis=1)
              compare = np.stack((QRESZ[:,vv], ZMAX),axis=1)
              QRESZ[:,vv] = bn.nanmin(compare, axis=1)
      
       return (QRESX, QRESZ)

def computeFlowVelocityCoeffs(U, W, DX, DZ, VSND):
                     
       QRESX = np.zeros((len(U), 4))
       QRESZ = np.zeros((len(W), 4))
       
       for vv in range(4):
              # Compute the anisotropic coefficients
              QRESX[:,vv] = (0.5 * DX) * U#(U + VSND)
              QRESZ[:,vv] = (0.5 * DZ) * W#(W + VSND)
       
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