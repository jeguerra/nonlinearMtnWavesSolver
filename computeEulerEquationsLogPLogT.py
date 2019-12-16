#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import numpy as np
import math as mt
import scipy.sparse as sps
import matplotlib.pyplot as plt

def computeInitialFields(PHYS, REFS, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex, m2):
       # Get some physical quantities
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       
       TQ = SOLT + INIT
       # Make the total quatities
       U = TQ[udex]
       LP = TQ[pdex]
       LT = TQ[tdex]
       
       # Compute the sensible temperature scaling to PGF
       RdT = Rd * P0**(-kap) * np.exp(LT + kap * LP)
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')
       
       # Update vertical velocity at the boundary
       dHdX = REFS[6]
       fields[botdex,1] = dHdX * np.array(U[botdex])
       
       # Make a smooth vertical decay for the input W
       ZTL = REFS[5]
       
       for kk in range(ZTL.shape[0]):
              zlev = ZTL[kk,:] - ZTL[0,:]
              # exponential decay by evanescent mode
              fields[botdex+kk,1] = np.array(fields[botdex,1]) * np.exp(m2 * zlev)
              
              #plt.plot(fields[botdex+kk,1])
              
       #plt.show()
       
       return fields, U, RdT

def computeUpdatedFields(PHYS, REFS, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex):
       # Get some physical quantities
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       
       TQ = SOLT + INIT
       # Make the total quatities
       U = TQ[udex]
       LP = TQ[pdex]
       LT = TQ[tdex]
       
       # Compute the sensible temperature scaling to PGF
       RdT = Rd * P0**(-kap) * np.exp(LT + kap * LP)
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')
       
       # Update vertical velocity at the boundary
       dHdX = REFS[6]
       fields[botdex,1] = dHdX * np.array(U[botdex])
       
       return fields, U, RdT

def computePrepareFields(PHYS, REFS, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex):
       # Get some physical quantities
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       
       TQ = SOLT + INIT
       # Make the total quatities
       U = TQ[udex]
       LP = TQ[pdex]
       LT = TQ[tdex]
       
       # Compute the sensible temperature scaling to PGF
       RdT = Rd * P0**(-kap) * np.exp(LT + kap * LP)
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')

       return fields, U, RdT

#%% Evaluate the Jacobian matrix
def computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, fields, U, RdT, botdex, topdex, isLinMFlux):
       # Switch nonlinearities in momentum flux on or off
       if isLinMFlux:
              NLS = 0.0
       else:
              NLS = 1.0
              
       # Get physical constants
       gc = PHYS[0]
       Rd = PHYS[3]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the derivative operators
       DDXM = REFS[10]
       DDZM = REFS[11]
       DZDX = REFS[15]
       
       # Compute terrain following terms (two way assignment into fields)
       wxz = np.array(fields[:,1])
       UZX = U * DZDX
       WXZ = wxz - UZX

       # Compute (total) derivatives of perturbations
       DqDx = DDXM.dot(fields)
       DqDz = DDZM.dot(fields)
       
       # Compute (partial) x derivatives of perturbations
       DZDXM = sps.diags(DZDX, offsets=0, format='csr')
       PqPx = DqDx - DZDXM.dot(DqDz)
       
       # Compute vertical gradient diagonal operators
       DuDzM = sps.diags(DqDz[:,0], offsets=0, format='csr')
       DwDzM = sps.diags(DqDz[:,1], offsets=0, format='csr')
       DlpDzM = sps.diags(DqDz[:,2], offsets=0, format='csr')
       DltDzM = sps.diags(DqDz[:,3], offsets=0, format='csr')
       
       # Compute horizontal gradient diagonal operators
       PuPxM = sps.diags(PqPx[:,0], offsets=0, format='csr')
       PwPxM = sps.diags(PqPx[:,1], offsets=0, format='csr')
       PlpPxM = sps.diags(PqPx[:,2], offsets=0, format='csr')
       PltPxM = sps.diags(PqPx[:,3], offsets=0, format='csr')
       
       # Compute hydrostatic state diagonal operators
       DLTDZ = REFG[1]
       DLTDZM = sps.diags(DLTDZ[:,0], offsets=0, format='csr')
       DQDZ = REFG[4]
       DUDZM = sps.diags(DQDZ[:,0], offsets=0, format='csr')
       DLPDZM = sps.diags(DQDZ[:,2], offsets=0, format='csr')
       DLPTDZM = sps.diags(DQDZ[:,3], offsets=0, format='csr')
       
       # Compute advective (multiplicative) diagonal operators
       UM = sps.diags(U, offsets=0, format='csr')
       WM = sps.diags(wxz, offsets=0, format='csr')
       WXZM = sps.diags(WXZ, offsets=0, format='csr')
       RdTM = sps.diags(RdT, offsets=0, format='csr')
       
       # Compute diagonal blocks related to sensible temperature
       RdT_bar = REFS[9]
       T_bar = (1.0 / Rd) * RdT_bar[:,0]
       T_ratio = np.exp(kap * fields[:,2] + fields[:,3]) - 1.0
       T_prime = T_ratio * T_bar
       RdT_barM = sps.diags(RdT_bar[:,0], offsets=0, format='csr')
       PtPx = DDXM.dot(T_prime) - DZDX * DDZM.dot(T_prime)
       DtDz = DDZM.dot(T_prime)
       PtPxM = sps.diags(PtPx, offsets=0, format='csr')
       DtDzM = sps.diags(DtDz, offsets=0, format='csr')
       
       # Compute partial in X terrain following block
       PPXM = DDXM - DZDXM.dot(DDZM)
       
       # Compute common horizontal transport block
       UPXM = UM.dot(DDXM) + WXZM.dot(DDZM)
       # Compute the vertical velocity transport block with nonlinearity switch
       UPXMW = UM.dot(PPXM) + NLS * (WM.dot(DDZM) + DwDzM)
       
       bf = sps.diags(T_ratio + 1.0, offsets=0, format='csr')
       
       # Compute the blocks of the Jacobian operator
       LD11 = UPXM + PuPxM
       LD12 = NLS * DuDzM + DUDZM
       LD13 = RdTM.dot(PPXM) + (Rd * PtPxM)
       LD14 = RdTM.dot(PlpPxM)
       
       LD21 = PwPxM + (1.0 - NLS) * DZDXM.dot(DwDzM) # NLS = 0.0 will cancel TF term
       LD22 = UPXMW
       LD23 = RdTM.dot(DDZM) + RdT_barM.dot(DLTDZM) + Rd * DtDzM
       LD24 = RdTM.dot(DlpDzM) - gc * bf
       
       LD31 = gam * PPXM + PlpPxM
       LD32 = gam * DDZM + DlpDzM + DLPDZM
       LD33 = UPXM
       LD34 = None
       
       LD41 = PltPxM
       LD42 = DltDzM + DLPTDZM
       LD43 = None
       LD44 = UPXM
       '''
       DDXBC = REFS[2]
       dHdX = REFS[6]
       # Compute coupled boundary adjustments
       UBC = sps.diags(U[botdex], offsets=0, format='csr')
       DuDxBC = sps.diags(DqDx[botdex,0], offsets=0, format='csr')
       ZDUDZBC = sps.diags(dHdX * DQDZ[botdex,0], offsets=0, format='csr')
       LD11[np.ix_(botdex,botdex)] = UBC.dot(DDXBC) + DuDxBC + ZDUDZBC
       
       WBC = sps.diags(dHdX * U[botdex], offsets=0, format='csr')
       ZDuDxBC = sps.diags(dHdX * DqDx[botdex,0], offsets=0, format='csr')
       ZDUDXBC = sps.diags(np.power(dHdX, 2.0) * DQDZ[botdex,0], offsets=0, format='csr')
       METBC = sps.diags(2.0 * U[botdex] * DDXBC.dot(dHdX), offsets=0, format='csr')
       LD21[np.ix_(botdex,botdex)] = WBC.dot(DDXBC) + ZDuDxBC + ZDUDXBC + METBC
       
       DlpDxBC = sps.diags(DqDx[botdex,2], offsets=0, format='csr')
       ZDLPDZBC = sps.diags(dHdX * DQDZ[botdex,2], offsets=0, format='csr')
       CONTBC = gam * DDXBC 
       LD31[np.ix_(botdex,botdex)] = DlpDxBC + ZDLPDZBC + CONTBC
       
       DlptDxBC = sps.diags(DqDx[botdex,3], offsets=0, format='csr')
       ZDLPTDZBC = sps.diags(dHdX * DQDZ[botdex,3], offsets=0, format='csr')
       LD41[np.ix_(botdex,botdex)] = DlptDxBC + ZDLPTDZBC
       '''
       DOPS = [LD11, LD12, LD13, LD14, \
               LD21, LD22, LD23, LD24, \
               LD31, LD32, LD33, LD34, \
               LD41, LD42, LD43, LD44]
       
       return DOPS

def computeJacobianVectorProduct(DOPS, REFG, vec, udex, wdex, pdex, tdex):
       # Get the Rayleight operators
       ROPS = REFG[5]
       
       # Compute the variable sections
       uvec = vec[udex]
       wvec = vec[wdex]
       pvec = vec[pdex]
       tvec = vec[tdex]
       
       # Compute the block products
       ures = (DOPS[0] + ROPS[0]).dot(uvec) + DOPS[1].dot(wvec) + DOPS[2].dot(pvec) + DOPS[3].dot(tvec)
       wres = DOPS[4].dot(uvec) + (DOPS[5] + ROPS[1]).dot(wvec) + DOPS[6].dot(pvec) + DOPS[7].dot(tvec)
       pres = DOPS[8].dot(uvec) + DOPS[9].dot(wvec) + (DOPS[10] + ROPS[2]).dot(pvec)
       tres = DOPS[12].dot(uvec) + DOPS[13].dot(wvec) + (DOPS[15] + ROPS[3]).dot(tvec)
       
       qprod = np.concatenate((ures, wres, pres, tres))
       
       return -qprod
    
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
       DUDZ = REFG[0]
       DLTDZ = REFG[1]
       DLPDZ = REFG[2]
       DLPTDZ = REFG[3]
       # Full spectral transform derivative matrices
       DDXM = REFS[10]
       DDZM = REFS[11]
       # Sparse 4th order compact FD derivative matrices
       #DDXM = REFS[12]
       #DDZM = REFS[13]
       DZDX = REFS[15]
              
       #%% Compute the various blocks needed
       UM = sps.spdiags(UZ[:,0], 0, OPS, OPS)
       PORZM = sps.spdiags(PORZ[:,0], 0, OPS, OPS)
       
       DUDZM = sps.spdiags(DUDZ[:,0], 0, OPS, OPS)
       DLTDZM = sps.spdiags(DLTDZ[:,0], 0, OPS, OPS)
       DLPDZM = sps.spdiags(DLPDZ[:,0], 0, OPS, OPS)
       DLPTDZM = sps.spdiags(DLPTDZ[:,0], 0, OPS, OPS)
       unit = sps.identity(OPS)
       
       DZDXM = sps.spdiags(DZDX, 0, OPS, OPS)
       
       #%% Compute the terms in the equations
       #U0DDX = UM.dot(DDXM)
       PPXM = DDXM - DZDXM.dot(DDZM)
       U0PPX = UM.dot(PPXM)
       
       # Horizontal momentum
       LD11 = U0PPX
       LD12 = DUDZM
       LD13 = PORZM.dot(PPXM)
       
       # Vertical momentum
       LD22 = U0PPX
       LD23 = PORZM.dot(DDZM + DLTDZM)
       # Equivalent form from direct linearization
       #LD23 = PORZM.dot(DDZM) + gc * (1.0 / gam - 1.0) * unit
       LD24 = -gc * unit
       
       # Log-P equation
       LD31 = gam * PPXM
       LD32 = gam * DDZM + DLPDZM
       LD33 = U0PPX
       
       # Log-Theta equation
       LD42 = DLPTDZM
       LD44 = U0PPX
       
       DOPS = [LD11, LD12, LD13, LD22, LD23, LD24, LD31, LD32, LD33, LD42, LD44]
       
       return DOPS

# Function evaluation of the non linear equations (dynamic components)
#@jit(nopython=True)
def computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, U, RdT, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the derivative operators
       DQDZ = REFG[4]
       DDXM = REFS[10]
       DDZM = REFS[11]
       DZDX = REFS[15]
       
       # Compute terrain following terms (two way assignment into fields)
       wxz = np.array(fields[:,1])
       UZX = U * DZDX
       WXZ = wxz - UZX
       
       # Compute advective (multiplicative) operators
       UM = sps.diags(U, offsets=0, format='csr')
       wxz = sps.diags(wxz, offsets=0, format='csr')
       WXZ = sps.diags(WXZ, offsets=0, format='csr')
       
       # Get the static horizontal and vertical derivatives
       wDQDZ = wxz.dot(DQDZ)
       
       # Compute derivative of perturbations
       DqDx = DDXM.dot(fields)
       DqDz = DDZM.dot(fields)
       
       # Compute advection
       UDqDx = UM.dot(DqDx)
       WDqDz = WXZ.dot(DqDz)
       transport = UDqDx + WDqDz + wDQDZ
       
       # Compute pressure gradient forces
       T_ratio = np.exp(kap * fields[:,2] + fields[:,3]) - 1.0
       PGFX = RdT * (DqDx[:,2] - DZDX * DqDz[:,2])
       PGFZ1 = RdT * (DqDz[:,2])
       PGFZ2 = -gc * T_ratio
       PGFZ = PGFZ1 + PGFZ2
       
       # Compute incompressibility constraint
       PuPx = np.array(DqDx[:,0] - DZDX * DqDz[:,0])
       PwPz = DqDz[:,1]
       incomp = gam * (PuPx + PwPz)

       def DqDt():
              # Horizontal momentum equation
              DuDt = -(transport[:,0] + PGFX)
              # Vertical momentum equation
              DwDt = -(transport[:,1] + PGFZ)
              # Pressure (mass) equation
              DpDt = -(transport[:,2] + incomp)
              # Potential Temperature equation
              DtDt = -(transport[:,3])
              
              # Make boundary adjustments
              DwDt[topdex] *= 0.0
              DwDt[botdex] *= 0.0
              DtDt[topdex] *= 0.0
              
              return (DuDt, DwDt, DpDt, DtDt)
                     
       return np.concatenate(DqDt())

def computeRayleighTendency(REFG, fields, botdex, topdex):
       
       # Get the Rayleight operators
       ROPS = REFG[5]
       
       # Compute the tendencies
       DuDt = - ROPS[0].dot(fields[:,0])
       DwDt = - ROPS[1].dot(fields[:,1])
       DpDt = - ROPS[2].dot(fields[:,2])
       DtDt = - ROPS[3].dot(fields[:,3])
       
       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt

def computeDynSGSTendency(RESCF, REFS, fields, udex, wdex, pdex, tdex, botdex, topdex):
       
       # Get the derivative operators
       DDXM = REFS[10]
       DDZM = REFS[11]
       #DDXM = REFS[12]
       #DDZM = REFS[13]
       DZDX = REFS[15]
       
       # Get the anisotropic coefficients
       RESCFX = RESCF[0]
       RESCFZ = RESCF[1]
       
       # Compute derivative of perturbations
       DDz = DDZM.dot(fields)
       DDx = DDXM.dot(fields)
       
       # Compute diffusive fluxes
       DuDx = RESCFX[udex] * (DDx[:,0] - DZDX * DDz[:,0])
       DwDx = RESCFX[wdex] * (DDx[:,1] - DZDX * DDz[:,1])
       DlpDx = RESCFX[pdex] * (DDx[:,2] - DZDX * DDz[:,2])
       DltDx = RESCFX[tdex] * (DDx[:,3] - DZDX * DDz[:,3])
       DuDz = RESCFZ[udex] * DDz[:,0]
       DwDz = RESCFZ[wdex] * DDz[:,1]
       DlpDz = RESCFZ[pdex] * DDz[:,2]
       DltDz = RESCFZ[tdex] * DDz[:,3]
       
       # Compute the tendencies (divergence of diffusive flux... discontinuous)
       #'''
       DuDt = DDXM.dot(DuDx) - DZDX * DDZM.dot(DuDx) + DDZM.dot(DuDz)
       DwDt = DDXM.dot(DwDx) - DZDX * DDZM.dot(DwDx) + DDZM.dot(DwDz)
       DpDt = DDXM.dot(DlpDx) - DZDX * DDZM.dot(DlpDx) + DDZM.dot(DlpDz)
       DtDt = DDXM.dot(DltDx) - DZDX * DDZM.dot(DltDx) + DDZM.dot(DltDz)
       #'''
       # Null tendencies along vertical boundaries
       DuDt[topdex] *= 0.0
       DwDt[topdex] *= 0.0
       DpDt[topdex] *= 0.0
       DtDt[topdex] *= 0.0

       DuDt[botdex] *= 0.0
       DwDt[botdex] *= 0.0
       DpDt[botdex] *= 0.0
       DtDt[botdex] *= 0.0

       # Concatenate
       DqDt = np.concatenate((DuDt, DwDt, DpDt, DtDt))
       
       return DqDt
       