#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np

def computeResidualViscCoeffs(RES, QM, U, W, DX, DZ):
       
       ARES = np.abs(RES)
              
       QRESX = 0.0 * RES
       QRESZ = 0.0 * RES
       
       for vv in range(4):
              
              if QM[vv] > 0.0:
                     # Compute the anisotropic coefficients
                     QRESX[:,vv] = (DX**2 / QM[vv]) * ARES[:,vv]
                     QRESZ[:,vv] = (DZ**2 / QM[vv]) * ARES[:,vv]
       
              # Fix SGS to upwind value where needed
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