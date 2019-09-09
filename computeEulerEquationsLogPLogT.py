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
       NX = DIMS[3] + 1
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
def computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, INIT, udex, wdex, pdex, tdex, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the derivative operators
       DDXM = REFS[13]
       DDZM = REFS[14]
       DZDX = REFS[15]
       
       # Get the static vertical gradients
       DUDZ = REFG[0]
       DLPDZ = REFG[1]
       DLPTDZ = REFG[2]
       
       # Make the total quatities
       U = np.add(uxz, INIT[udex])
       LP = np.add(pxz, INIT[pdex])
       LT = np.add(txz, INIT[tdex])
        
       # Compute the sensible temperature scaling to PGF
       RdT = Rd * P0**(-kap) * np.exp(LT + kap * LP)
       
       # Compute derivative of perturbations
       DuDx = DDXM.dot(uxz)
       DuDz = DDZM.dot(uxz)
       DwDx = DDXM.dot(wxz)
       DwDz = DDZM.dot(wxz)
       DlpDx = DDXM.dot(pxz)
       DlpDz = DDZM.dot(pxz)
       DltDx = DDXM.dot(txz)
       DltDz = DDZM.dot(txz)
       
       # Terrain following horizontal derivatives
       DUDX = DuDx - DZDX * (DuDz + DUDZ)
       DWDX = DwDx - DZDX * DwDz
       DLPDX = DlpDx - DZDX * (DlpDz + DLPDZ)
       DLTDX = DltDx - DZDX * (DltDz + DLPTDZ)
       
       # Horizontal Momentum
       LD11 = U * DUDX
       LD12nl = wxz * DuDz
       LD12ln = wxz * DUDZ
       LD13 = RdT * DLPDX
       
       # Vertical Momentum
       LD21 = U * DWDX
       LD22 = wxz * DwDz
       LD23 = RdT * (DlpDz + DLPDZ) + gc
       
       # Pressure (mass) equation
       LD31 = U * DLPDX
       LD32nl = (wxz * DlpDz)
       LD32ln = wxz * DLPDZ
       LD33 = gam * DUDX
       LD34 = gam * DwDz
       
       # Potential Temperature equation
       LD41 = U * DLTDX
       LD42nl = (wxz * DltDz)
       LD42ln = (wxz * DLPTDZ)
       
       # No transport of horizontal momentum or entropy to the boundary
       #LD12nl[botdex] = np.zeros(len(botdex))
       #LD42nl[botdex] = np.zeros(len(botdex))
       #LD12nl[topdex] = np.zeros(len(botdex))
       #LD42nl[topdex] = np.zeros(len(botdex))

       # Compute tendency for semilinear terms
       DuDt = -(LD11 + LD12ln + LD12nl + LD13)
       DwDt = -(LD21 + LD22 + LD23)
       DpDt = -(LD31 + LD32ln + LD32nl + LD33 + LD34)
       DtDt = -(LD41 + LD42ln + LD42nl)
       
       # Apply BC to the tendency
       DwDt[botdex] = np.zeros(len(botdex))
       DuDt[topdex] = np.zeros(len(topdex))
       DwDt[topdex] = np.zeros(len(topdex))
       DtDt[topdex] = np.zeros(len(topdex))
       
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt #_ln[sysDex], DqDt_nl[sysDex]

def computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex):
       
       # Get the static vertical gradients
       ROPS = REFG[3]
       
       # Compute the tendencies
       DuDt = - ROPS[0].dot(uxz)
       DwDt = - ROPS[1].dot(wxz)
       DpDt = - ROPS[2].dot(pxz)
       DtDt = - ROPS[3].dot(txz)
       
       # Apply BC to the tendency
       DwDt[botdex] = np.zeros(len(botdex))
       DuDt[topdex] = np.zeros(len(topdex))
       DwDt[topdex] = np.zeros(len(topdex))
       DtDt[topdex] = np.zeros(len(topdex))
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt

def computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex):
       
       # Get the derivative operators
       DDXM = REFS[13]
       DDZM = REFS[14]
       
       # Get the anisotropic coefficients
       RESCFX = RESCF[0]
       RESCFZ = RESCF[1]
       
       # Compute the tendencies
       DuDt = DDXM.dot(RESCFX[udex] * DDXM.dot(uxz)) + DDZM.dot(RESCFZ[udex] * DDZM.dot(uxz))
       DwDt = DDXM.dot(RESCFX[wdex] * DDXM.dot(wxz)) + DDZM.dot(RESCFZ[wdex] * DDZM.dot(wxz))
       DpDt = DDXM.dot(RESCFX[pdex] * DDXM.dot(pxz)) + DDZM.dot(RESCFZ[pdex] * DDZM.dot(pxz))
       DtDt = DDXM.dot(RESCFX[tdex] * DDXM.dot(txz)) + DDZM.dot(RESCFZ[tdex] * DDZM.dot(txz))
       
       # Apply BC to the tendency
       DwDt[botdex] = np.zeros(len(botdex))
       DuDt[topdex] = np.zeros(len(topdex))
       DwDt[topdex] = np.zeros(len(topdex))
       DtDt[topdex] = np.zeros(len(topdex))
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt
       