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
def computeResidualViscCoeffs(RES, QM, VFLW, DX, DZ, DX2, DZ2, RHOI):
       
       # Turn off residual coefficients in the sponge layers
       ARES = np.abs(RES)
       
       # Normalize the residuals (U and W only!)
       for vv in range(2):
              if QM[vv] > 0.0:
                     ARES[:,vv] *= (1.0 / QM[vv])
              else:
                     ARES[:,vv] *= 0.0
                     
       # Get the maximum in the residuals (unit = 1/s)
       QRES_MAX =  bn.nanmax(ARES, axis=1)
       
       # Upper bound for coefficients (unit = 1/s)
       loclen = mt.sqrt(DX * DZ)
       QMAX = 0.5 / loclen * VFLW
       
       # Limit DynSGS to upper bound
       compare = np.stack((QRES_MAX, QMAX),axis=1)
       QRES_CF = bn.nanmin(compare, axis=1)

       return (np.expand_dims(RHOI * QRES_CF,1), np.expand_dims(RHOI * QMAX,1))

def computeFlowAccelerationCoeffs(RES, DT, U, W, DX, DZ):
       
       ARES = np.abs(RES)
              
       QRESX = np.zeros((len(U), 4))
       QRESZ = np.zeros((len(W), 4))
       
       for vv in range(4):
              # Compute the anisotropic coefficients
              QRESX[:,vv] = (DX * DT) * ARES[0,vv]
              QRESZ[:,vv] = (DZ * DT) * ARES[1,vv]

       return (QRESX, QRESZ)