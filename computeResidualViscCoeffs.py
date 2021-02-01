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
def computeResidualViscCoeffs(RES, QM, VFLW, DX, DZ, DXD, DZD, DX2, DZ2):
       
       # Compute a filter length...
       DXZ = DXD * DZD
       DL = mt.sqrt(DXZ)
       
       # Compute absolute value of residuals
       ARES = np.abs(RES)
       
       # Normalize the residuals (U and W only!)
       for vv in range(2):
              if QM[vv] > 0.0:
                     ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX = DXZ * bn.nanmax(ARES, axis=1)
       
       # Compute flow speed plus sound speed coefficients
       QMAX = 0.5 * DL * VFLW
       
       # Limit DynSGS to upper bound
       compare = np.stack((QRES_MAX, QMAX),axis=1)
       QRES_CF = bn.nanmin(compare, axis=1)

       return (np.expand_dims(QRES_CF,1), np.expand_dims(QMAX,1))

def computeCellAveragingOperator(DIMS):
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Initialize matrix
       CAM = np.zeros((OPS, OPS))
       
       # Compute edge indices
       bdex = np.array(range(0, OPS, NZ))
       tdex = np.array(range(NZ-1, OPS, NZ))
       ldex = np.array(range(bdex[0], NZ))
       rdex = np.array(range(bdex[-1], OPS))
       
       # Compute interior indices
       adex = np.array(range(OPS))
       
       rowsAll = set(adex)
       rowsInt = rowsAll.difference(set(np.concatenate(bdex, ldex, tdex, rdex)))
       '''
       # Loop over the rows (2D FORTRAN ORDER TARGET VECTOR)
       stencil = []
       for ii in ldex:
              loc = range()
              CAM[ii,]
       '''      
       return CAM
