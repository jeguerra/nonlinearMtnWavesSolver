#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np

def computeResidualViscCoeffs(SOL, RES, DX, DZ, udex, OPS):
       
       ARES = np.abs(RES)
       DSOL = np.abs(SOL)
       
       QRESX = np.zeros(SOL.shape)
       QRESZ = np.zeros(SOL.shape)
       
       for vv in range(4):
              qdex = udex + vv * OPS
              # Get the normalization from the current estimate
              QM = np.amax(DSOL[qdex])
              
              if QM > 0.0:
                     # Compute the anisotropic coefficients
                     QRESX[qdex] = (DX**2 / QM) * ARES[qdex]
                     QRESZ[qdex] = (DZ**2 / QM) * ARES[qdex]
       
       # Fix SGS to upwind value where needed
       updex = np.argwhere(QRESX >= 0.5 * DX)
       QRESX[updex] = 0.5 * DX * np.ones((len(updex),1))
       updex = np.argwhere(QRESZ >= 0.5 * DZ)
       QRESZ[updex] = 0.5 * DZ * np.ones((len(updex),1))
       
       return (QRESX, QRESZ)