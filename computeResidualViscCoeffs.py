#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np

def computeResidualViscCoeffs(RES, QM, U, W, DX, DZ):
       
       ARES = np.abs(RES)
              
       # Compute the anisotropic coefficients
       QRESX = (DX**2 * ARES)
       QRESZ = (DZ**2 * ARES)
       
       # Fix SGS to upwind value where needed
       for vv in range(4):
              
              if QM[vv] > 0.0:
                     QRESX[:,vv] /= QM[vv]
              else:
                     QRESX[:,vv] *= 0.0
                     
              fdex = QRESX[:,vv] > (0.5 * DX) * U
              QRESX[fdex, vv] = (0.5 * DX) * U[fdex]
              fdex = QRESZ[:,vv] > (0.5 * DZ) * W
              QRESZ[fdex, vv] = (0.5 * DZ) * W[fdex]

       return (QRESX, QRESZ)

def computeFlowVelocityCoeffs(U, W, DX, DZ):
                     
       QRESX = np.zeros((len(U), 4))
       QRESZ = np.zeros((len(W), 4))
       
       for vv in range(4):
              # Compute the anisotropic coefficients
              QRESX[:,vv] = (0.5 * DX) * U
              QRESZ[:,vv] = (0.5 * DZ) * W
       
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