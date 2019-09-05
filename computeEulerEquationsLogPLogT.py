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
def computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, INIT, sysDex, udex, wdex, pdex, tdex, bdex):
       # Get physical constants
       gc = PHYS[0]
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the derivative operators
       DDXM = REFS[13]
       DDZM = REFS[14]
       
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
        
       # Horizontal Momentum
       LD11 = U * DuDx
       LD12nl = wxz * DuDz
       LD12ln = wxz * DUDZ
       LD13 = RdT * DlpDx
       
       # Vertical Momentum
       LD21 = U * DwDx
       LD22 = wxz * DwDz
       LD23 = RdT * (DlpDz + DLPDZ) + gc
       
       # Pressure (mass) equation
       LD31 = U * DlpDx
       LD32nl = (wxz * DlpDz)
       LD32ln = wxz * DLPDZ
       LD33 = gam * DuDx
       LD34 = gam * DwDz
       
       # Potential Temperature equation
       LD41 = U * DltDx
       LD42nl = (wxz * DltDz)
       LD42ln = (wxz * DLPTDZ)

       # Compute tendency for semilinear terms
       DqDt_ln = np.zeros(4 * len(udex))
       DqDt_ln[udex] = -(LD11 + LD12ln + LD13)
       DqDt_ln[wdex] = -(LD21 + LD22 + LD23)
       DqDt_ln[pdex] = -(LD31 + LD32ln + LD33 + LD34)
       DqDt_ln[tdex] = -(LD41 + LD42ln)
       
       # Compute tendency for nonlinear terms (subcycle)
       DqDt_nl = np.zeros(4 * len(udex))
       DqDt_nl[udex] = -(LD12nl)
       #DqDt_nl[wdex] = -(add nonlinear bouyancy terms if necessary)
       DqDt_nl[pdex] = -(LD32nl)
       DqDt_nl[tdex] = -(LD42nl)
       
       return DqDt_ln[sysDex], DqDt_nl[sysDex]

def computeRayleighTendency(REFG, uxz, wxz, pxz, txz, sysDex, udex, wdex, pdex, tdex, bdex):
       
       # Get the static vertical gradients
       ROPS = REFG[3]
       
       # Compute the tendencies
       DuDt = - ROPS[0].dot(uxz)
       DwDt = - ROPS[1].dot(wxz)
       DpDt = - ROPS[2].dot(pxz)
       DtDt = - ROPS[3].dot(txz)
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt[sysDex]

def computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, sysDex, udex, wdex, pdex, tdex, bdex):
       
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
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt[sysDex]
       