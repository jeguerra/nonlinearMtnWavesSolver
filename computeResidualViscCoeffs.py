#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np

def computeResidualViscCoeffs(SOL, RES, DX, DZ, udex, wdex, pdex, tdex):
       
       ARES = np.abs(RES)
       DSOL = np.abs(SOL)
              
       QRESX = 0.0 * SOL
       QRESZ = 0.0 * SOL
       
       for vv in range(4):
              
              if vv == 0:
                     qdex = udex
              elif vv == 1:
                     qdex = wdex
              elif vv == 2:
                     qdex = pdex
              elif vv == 3:
                     qdex = tdex
                     
              # Get the normalization from the current estimate
              QM = np.amax(DSOL[qdex])
              
              if QM > 0.0:
                     # Compute the anisotropic coefficients
                     QRESX[qdex] = (DX**2 / QM) * ARES[qdex]
                     QRESZ[qdex] = (DZ**2 / QM) * ARES[qdex]
       
       # Fix SGS to upwind value where needed
       #'''
       updex = np.argwhere(QRESX >= 0.5 * DX)
       QRESX[updex] = 0.5 * DX
       updex = np.argwhere(QRESZ >= 0.5 * DZ)
       QRESZ[updex] = 0.5 * DZ
       #'''
       return (QRESX, QRESZ)