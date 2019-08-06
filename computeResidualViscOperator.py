#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np
import scipy.sparse as sps

def computeResidualViscOperator(DIMS, REFS, RES, qdex, DX, DZ):
       # Get the dimensions
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = NX * NZ
        
       # Get REFS data
       DDXM = REFS[13]
       DDZM = REFS[14]
       DDXM2 = REFS[15]
       DDZM2 = REFS[16]
       
       # Get the normalization from the current estimate
       QM = np.amax(RES[qdex,0])
       
       # Compute the anisotropic coefficients
       QRESX = (DX**2 / QM) * np.abs(RES[qdex,2])
       QRESZ = (DZ**2 / QM) * np.abs(RES[qdex,2])
       
       # Fix SGS to upwind value where needed
       updex = np.argwhere(QRESX >= 0.5 * DX)
       QRESX[updex] = 0.5 * DX * np.ones((len(updex),1))
       updex = np.argwhere(QRESZ >= 0.5 * DZ)
       QRESZ[updex] = 0.5 * DZ * np.ones((len(updex),1))
       
       # Make diagonal operators
       QRESX = sps.spdiags(QRESX, 0, OPS, OPS)
       QRESZ = sps.spdiags(QRESZ, 0, OPS, OPS)       
       
       # Operator for this quantity
       #term1 = DDXM.dot(QRESX.dot(DDXM))
       #term2 = DDZM.dot(QRESZ.dot(DDZM))
       term1 = QRESX.dot(DDXM2)
       term2 = QRESZ.dot(DDZM2)
       RVDQ = term1 + term2
       
       return RVDQ