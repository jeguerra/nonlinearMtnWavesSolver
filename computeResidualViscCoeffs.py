#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np

def computeResidualViscCoeffs(fields, RES, QM, DX, DZ):
       
       ARES = np.abs(RES)
       DSOL = np.abs(fields)
              
       QRESX = 0.0 * RES
       QRESZ = 0.0 * RES
       
       for vv in range(4):
              
              if QM[vv] > 0.0:
                     # Compute the anisotropic coefficients
                     QRESX[:,vv] = (DX**2 / QM[vv]) * ARES[:,vv]
                     QRESZ[:,vv] = (DZ**2 / QM[vv]) * ARES[:,vv]
       
              # Fix SGS to upwind value where needed
              updex = np.argwhere(QRESX[:,vv] >= 0.5 * DX)
              QRESX[updex,vv] = 0.5 * DX
              updex = np.argwhere(QRESZ[:,vv] >= 0.5 * DZ)
              QRESZ[updex,vv] = 0.5 * DZ
       
       return (QRESX, QRESZ)