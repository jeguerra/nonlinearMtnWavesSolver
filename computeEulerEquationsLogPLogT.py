#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import numpy as np
import scipy.sparse as sps

#%% The linear equation operator
def computeEulerEquationsLogPLogT(DIMS, PHYS, REFS, REFG):
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
       DUDZ = REFG[3]
       DLPDZ = REFG[4]
       DLPTDZ = REFG[5]
       DDXM = REFS[10]
       DDZM = REFS[11]
       DZDX = REFS[15]
              
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
       tempDiagonal = np.reshape(DZDX, (OPS,), order='F')
       DZDXM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       unit = sps.identity(OPS)
       
       #%% Compute the terms in the equations
       PPX = DDXM - DZDXM.dot(DDZM)
       U0DDX = UM.dot(PPX)
       
       # Horizontal momentum
       LD11 = U0DDX
       LD12 = DUDZM
       LD13 = PORZM.dot(PPX) - gc * (1.0 / gam - 1.0) * DZDXM
       LD14 = gc * DZDXM
       
       # Vertical momentum
       LD22 = U0DDX
       LD23 = PORZM.dot(DDZM) + gc * (1.0 / gam - 1.0) * unit
       LD24 = -gc * unit
       
       # Log-P equation
       LD31 = gam * PPX
       LD32 = gam * DDZM + DLPDZM
       LD33 = U0DDX
       
       # Log-Theta equation
       LD42 = DLPTDZM
       LD44 = U0DDX
       
       DOPS = [LD11, LD12, LD13, LD14, LD22, LD23, LD24, LD31, LD32, LD33, LD42, LD44]
       
       return DOPS

# Function evaluation of the non linear equations (dynamic components)
def computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, uxz, wxz, pxz, txz, U, RdT, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       gam = PHYS[6]
       
       # Get the derivative operators
       DDXM = REFS[10]
       DDZM = REFS[11]
       DZDX = REFS[15]
       
       # Get the static horizontal and vertical derivatives
       DUDZ = REFG[3]
       DLPDZ = REFG[4]
       DLPTDZ = REFG[5]
       
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
       
       # Compute terrain following terms
       WXZ = wxz - U * DZDX
       PlpPz = DlpDz + DLPDZ
       PGFZ = RdT * PlpPz + gc
       
       # Horizontal momentum equation
       LD11 = U * DuDx
       LD12 = WXZ * DuDz + wxz * DUDZ
       LD13 = RdT * DlpDx - DZDX * PGFZ
       DuDt = -(LD11 + LD12 + LD13)
       # Vertical momentum equation
       LD21 = U * DwDx
       LD22 = WXZ * DwDz
       LD23 = PGFZ
       DwDt = -(LD21 + LD22 + LD23)
       # Pressure (mass) equation
       LD31 = U * DlpDx
       LD32 = WXZ * DlpDz + wxz * DLPDZ
       LD33 = gam * (DuDx + (1.0 - DZDX) * DwDz)
       #LD33 = gam * (DuDx - DZDX * DwDz + DwDz)
       DpDt = -(LD31 + LD32 + LD33)
       # Potential Temperature equation
       LD41 = U * DltDx
       LD42 = WXZ * DltDz + wxz * DLPTDZ
       DtDt = -(LD41 + LD42)
       
       DwDt[topdex] *= 0.0
       DwDt[botdex] *= 0.0
       DtDt[topdex] *= 0.0
       
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt

def computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex):
       
       # Get the static vertical gradients
       ROPS = REFG[6]
       
       # Compute the tendencies
       DuDt = - ROPS[0].dot(uxz)
       DwDt = - ROPS[1].dot(wxz)
       DpDt = - ROPS[2].dot(pxz)
       DtDt = - ROPS[3].dot(txz)
       
       # Null tendencies at essential vertical boundaries
       DuDt[topdex] *= 0.0
       DuDt[botdex] *= 0.0
       DwDt[topdex] *= 0.0
       DwDt[botdex] *= 0.0
       DpDt[topdex] *= 0.0
       DpDt[botdex] *= 0.0
       DtDt[topdex] *= 0.0
       DtDt[botdex] *= 0.0
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt

def computeDynSGSTendency(RESCF, REFS, fields, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex):
       
       # Get the derivative operators
       #DDXM = REFS[10]
       #DDZM = REFS[11]
       DDXM2 = REFS[12]
       DDZM2 = REFS[13]
       
       # Get the anisotropic coefficients
       RESCFX = RESCF[0]
       RESCFZ = RESCF[1]
       
       # Compute derivative of perturbations
       DDx = DDXM2.dot(fields)
       DDz = DDZM2.dot(fields)
       '''
       DuDx = DDx[:,0]
       DwDx = DDx[:,1]
       DlpDx = DDx[:,2]
       DltDx = DDx[:,3]
       DuDz = DDz[:,0]
       DwDz = DDz[:,1]
       DlpDz = DDz[:,2]
       DltDz = DDz[:,3]
       '''
       # Compute the tendencies (divergence of diffusive flux... discontinuous)
       '''
       DuDt = DDXM.dot(RESCFX[udex] * DuDx) + DDZM.dot(RESCFZ[udex] * DuDz)
       DwDt = DDXM.dot(RESCFX[wdex] * DwDx) + DDZM.dot(RESCFZ[wdex] * DwDz)
       DpDt = DDXM.dot(RESCFX[pdex] * DlpDx) + DDZM.dot(RESCFZ[pdex] * DlpDz)
       DtDt = DDXM.dot(RESCFX[tdex] * DltDx) + DDZM.dot(RESCFZ[tdex] * DltDz)
       '''
       # Compute tendencies (2nd derivative term only)
       #'''
       DuDt = RESCFX[udex] * DDx[:,0] + RESCFZ[udex] * DDz[:,0]
       DwDt = RESCFX[wdex] * DDx[:,1] + RESCFZ[wdex] * DDz[:,1]
       DpDt = RESCFX[pdex] * DDx[:,2] + RESCFZ[pdex] * DDz[:,2]
       DtDt = RESCFX[tdex] * DDx[:,3] + RESCFZ[tdex] * DDz[:,3]
       #'''
       # Null tendencies along vertical boundaries
       DuDt[topdex] *= 0.0
       DuDt[botdex] *= 0.0
       DwDt[topdex] *= 0.0
       DwDt[botdex] *= 0.0
       DpDt[topdex] *= 0.0
       DpDt[botdex] *= 0.0
       DtDt[topdex] *= 0.0
       DtDt[botdex] *= 0.0

       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt
       