#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:59:02 2019

@author: TempestGuerra
"""

import numpy as np

def computeResidualViscCoeffs(RES, qdex, DX, DZ, DDXM, DDZM):
       
       ARES = np.abs(RES[qdex,2])
       DSOL = np.abs(RES[qdex,0])
       
       # Get the normalization from the current estimate
       QM = np.amax(DSOL)
       
       # Compute the anisotropic coefficients
       QRESX = (DX**2 / QM) * ARES
       QRESZ = (DZ**2 / QM) * ARES
       
       # Fix SGS to upwind value where needed
       updex = np.argwhere(QRESX >= 0.5 * DX)
       QRESX[updex] = 0.5 * DX * np.ones((len(updex),1))
       updex = np.argwhere(QRESZ >= 0.5 * DZ)
       QRESZ[updex] = 0.5 * DZ * np.ones((len(updex),1))
       
       # Compute the SGS stress estimates
       TauX = QRESX * DDXM.dot(RES[qdex,0])
       TauZ = QRESZ * DDZM.dot(RES[qdex,0])
       
       # Compute divergence of the stress
       DynSGSX = DDXM.dot(TauX)
       DynSGSZ = DDZM.dot(TauZ)
       
       # Compute the damping tendency
       dqdt = DynSGSX + DynSGSZ
       
       return dqdt