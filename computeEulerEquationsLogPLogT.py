#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""

import numpy as np
import scipy.sparse as sps
#import matplotlib.pyplot as plt

#%% The linear equation operator
def computeEulerEquationsLogPLogT(DIMS, PHYS, REFS):
       # Get physical constants
       gc = PHYS[0]
       gam = PHYS[6]
       
       # Get the dimensions
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Get REFS data
       UZ = REFS[8]
       PORZ = REFS[9]
       DUDZ = REFS[10]
       DLPDZ = REFS[11]
       DLPTDZ = REFS[12]
       DDXM = REFS[13]
       DDZM = REFS[14]
              
       #%% Compute the various blocks needed
       tempDiagonal = np.reshape(UZ, (OPS,), order='F')
       UM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       tempDiagonal = np.reshape(DUDZ, (OPS,), order='F')
       DUDZM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       tempDiagonal = np.reshape(DLPDZ, (OPS,), order='F')
       DLPDZM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       tempDiagonal = np.reshape(DLPTDZ, (OPS,), order='F')
       DLPTDZM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       U0DX = UM.dot(DDXM)
       tempDiagonal = np.reshape(PORZ, (OPS,), order='F')
       PORZM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       unit = sps.identity(OPS)
       
       #%% Compute the terms in the equations
       
       # Horizontal momentum
       LD11 = U0DX
       LD12 = DUDZM
       LD13 = PORZM.dot(DDXM)
       
       # Vertical momentum
       LD22 = U0DX
       LD23 = PORZM.dot(DDZM) + gc * (1.0 / gam - 1.0) * unit
       LD24 = -gc * unit
       
       # Log-P equation
       LD31 = gam * DDXM
       LD32 = gam * DDZM + DLPDZM
       LD33 = U0DX
       
       # Log-Theta equation
       LD42 = DLPTDZM
       LD44 = U0DX
       
       DOPS = [LD11, LD12, LD13, LD22, LD23, LD24, LD31, LD32, LD33, LD42, LD44]
       
       return DOPS

# Function evaluation of the non linear equations
def computeEulerEquationsLogPLogT_NL(DIMS, PHYS, REFS, uxz, wxz, pxz, txz):
       # Get physical constants
       gc = PHYS[0]
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the dimensions
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Get the derivative operators
       DDXM = REFS[13]
       DDZM = REFS[14]
       
       #%% Compute the terms in the equations
       
       # Compute the sensible temperature
       RdT = Rd * P0**(-kap) * np.exp(txz + kap * pxz)
        
       # Horizontal Momentum
       LD11 = 0.5 * DDXM.dot(np.power(uxz, 2.0))
       LD12 = np.multiply(wxz, DDZM.dot(uxz))
       LD13 = np.multiply(RdT, DDXM.dot(pxz))
       
       # Vertical Momentum
       LD21 = 0.5 * DDZM.dot(np.power(wxz, 2.0))
       LD22 = np.multiply(uxz, DDXM.dot(wxz))
       LD23 = np.multiply(RdT, DDZM.dot(pxz)) + gc * sps.spdiags(np.ones((OPS,)), 0, OPS, OPS)
       
       # Pressure (mass) equation
       
       # Potential Temperature equation
       
       return RHSU, RHSW, RHSP, RHST
       