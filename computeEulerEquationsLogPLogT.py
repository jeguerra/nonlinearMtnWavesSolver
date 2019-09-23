#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import numpy as np
import scipy.sparse as sps

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
       tempDiagonal = np.reshape(PORZ, (OPS,), order='F')
       PORZM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       unit = sps.identity(OPS)
       
       #%% Compute the terms in the equations
       U0DDX = UM.dot(DDXM)
       
       # Horizontal momentum
       LD11 = U0DDX
       LD12 = DUDZM
       LD13 = PORZM.dot(DDXM)
       
       # Vertical momentum
       LD22 = U0DDX
       LD23 = PORZM.dot(DDZM) + gc * (1.0 / gam - 1.0) * unit
       LD24 = -gc * unit
       
       # Log-P equation
       LD31 = gam * DDXM
       LD32 = gam * DDZM + DLPDZM
       LD33 = U0DDX
       
       # Log-Theta equation
       LD42 = DLPTDZM
       LD44 = U0DDX
       
       DOPS = [LD11, LD12, LD13, LD22, LD23, LD24, LD31, LD32, LD33, LD42, LD44]
       
       return DOPS

# Function evaluation of the non linear equations
def computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, uxz, wxz, pxz, txz, U, RdT, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       gam = PHYS[6]
       
       # Get the derivative operators
       DDXM = REFS[13]
       DDZM = REFS[14]
       
       # Get the static vertical gradients
       DUDZ = REFG[0]
       DLPDZ = REFG[1]
       DLPTDZ = REFG[2]
       
       # Compute derivative of perturbations
       DDx = DDXM.dot(fields)
       DDz = DDZM.dot(fields)
       DuDx = DDx[:,0]
       DwDx = DDx[:,1]
       DlpDx = DDx[:,2]
       DltDx = DDx[:,3]
       DuDz = DDz[:,0]
       DwDz = DDz[:,1]
       DlpDz = DDz[:,2]
       DltDz = DDz[:,3]
       
       # Heat and momentum fluxes
       #uw = uxz * wxz
       #DuwDz = DDZM.dot(uw)
       #tw = txz * wxz
       #DtwDz = DDZM.dot(tw)
       
       # Horizontal momentum equation
       LD11 = U * DuDx
       LD12 = wxz * (DuDz + DUDZ)
       LD13 = RdT * DlpDx
       DuDt = -(LD11 + LD12 + LD13)
       # Vertical momentum equation
       LD21 = U * DwDx
       LD22 = wxz * DwDz
       LD23 = RdT * (DlpDz + DLPDZ) + gc
       DwDt = -(LD21 + LD22 + LD23)
       # Pressure (mass) equation
       LD31 = U * DlpDx
       LD32 = wxz * (DlpDz + DLPDZ)
       LD33 = gam * (DuDx + DwDz)
       DpDt = -(LD31 + LD32 + LD33) 
       # Potential Temperature equation
       LD41 = U * DltDx
       LD42 = wxz * (DltDz + DLPTDZ)
       DtDt = -(LD41 + LD42)
       
       DwDt[topdex] *= 0.0
       DwDt[botdex] *= 0.0
       DtDt[topdex] *= 0.0
       
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt

def computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex):
       
       # Get the static vertical gradients
       ROPS = REFG[3]
       
       # Compute the tendencies
       DuDt = - ROPS[0].dot(uxz)
       DwDt = - ROPS[1].dot(wxz)
       DpDt = - ROPS[2].dot(pxz)
       DtDt = - ROPS[3].dot(txz)
       
       # Null tendencies at vertical boundaries
       DwDt[topdex] *= 0.0
       DwDt[botdex] *= 0.0
       DtDt[topdex] *= 0.0
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt

def computeDynSGSTendency(RESCF, REFS, fields, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex):
       
       # Get the derivative operators
       #DDXM = REFS[13]
       #DDZM = REFS[14]
       DDXM2 = REFS[15]
       DDZM2 = REFS[16]
       
       # Get the anisotropic coefficients
       RESCFX = RESCF[0]
       RESCFZ = RESCF[1]
       
       # Compute derivative of perturbations
       DDx = DDXM2.dot(fields)
       DDz = DDZM2.dot(fields)
       DuDx = DDx[:,0]
       DwDx = DDx[:,1]
       DlpDx = DDx[:,2]
       DltDx = DDx[:,3]
       DuDz = DDz[:,0]
       DwDz = DDz[:,1]
       DlpDz = DDz[:,2]
       DltDz = DDz[:,3]
       
       # Compute the tendencies (divergence of diffusive flux... discontinuous)
       #DuDt = DDXM.dot(RESCFX[udex] * DDXM.dot(uxz)) + DDZM.dot(RESCFZ[udex] * DDZM.dot(uxz))
       #DwDt = DDXM.dot(RESCFX[wdex] * DDXM.dot(wxz)) + DDZM.dot(RESCFZ[wdex] * DDZM.dot(wxz))
       #DpDt = DDXM.dot(RESCFX[pdex] * DDXM.dot(pxz)) + DDZM.dot(RESCFZ[pdex] * DDZM.dot(pxz))
       #DtDt = DDXM.dot(RESCFX[tdex] * DDXM.dot(txz)) + DDZM.dot(RESCFZ[tdex] * DDZM.dot(txz))
       
       # Compute tendencies (2nd derivative term only)
       DuDt = RESCFX[udex] * DuDx + RESCFZ[udex] * DuDz
       DwDt = RESCFX[wdex] * DwDx + RESCFZ[wdex] * DwDz
       DpDt = RESCFX[pdex] * DlpDx + RESCFZ[pdex] * DlpDz
       DtDt = RESCFX[tdex] * DltDx + RESCFZ[tdex] * DltDz
       
       # Null tendencies at vertical boundaries
       #DuDt[topdex] *= 0.0
       #DuDt[botdex] *= 0.0
       DwDt[topdex] *= 0.0
       DwDt[botdex] *= 0.0
       #DpDt[topdex] *= 0.0
       #DpDt[botdex] *= 0.0
       DtDt[topdex] *= 0.0
       #DtDt[botdex] *= 0.0
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt
       