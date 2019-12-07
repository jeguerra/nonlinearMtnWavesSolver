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

def computeInitialFields(PHYS, REFS, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex):
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
       #plt.figure()
       #plt.plot(fields[botdex,1])
       # Make a smooth vertical decay for the input W
       ZTL = REFS[5]
       zeroLev = 10
       p = 4
       DZ = ZTL[zeroLev,:] - ZTL[0,:]
       for kk in range(1,zeroLev+1):
              normZlev = ZTL[kk,:] * np.reciprocal(DZ)
              # Polynomial decay
              fields[botdex+kk,1] = np.power(normZlev - 1.0, p) * \
                                    np.array(fields[botdex,1])
              # Polynomial cosine decay
              #fields[botdex+kk,1] = np.power(np.cos(0.5 * mt.pi * normZlev), p) * \
              #                      np.array(fields[botdex,1])
              
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
def computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, fields, U, RdT, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       Rd = PHYS[3]
       gam = PHYS[6]
       
       # Get the derivative operators
       DDXM = REFS[10]
       DDZM = REFS[11]
       DZDX = REFS[15]
       
       # Compute terrain following terms
       wxz = fields[:,1]
       WXZ = wxz - U * DZDX
       
       # Compute boundary term adjustments
       UBC = np.array(0.0 * U)
       WBC = np.array(0.0 * wxz)
       UBC[botdex] = np.array(U[botdex])
       WBC[botdex] = np.array(wxz[botdex])
       UBCM = sps.diags(UBC, offsets=0, format='csr')
       WBCM = sps.diags(WBC, offsets=0, format='csr')
       
       WXZ[botdex] *= 0.0
       
       DZDX_MT = np.array(DZDX)
       DZDX_MT[botdex] *= 0.0
       DZDXM_MT = sps.diags(DZDX_MT, offsets=0, format='csr')
       
       DZDX_BC = 0.0 * np.array(DZDX)
       DZDX_BC[botdex] = np.array(DZDX[botdex])
       DZDXM_BC = sps.diags(DZDX_BC, offsets=0, format='csr')

       # Compute (total) derivatives of perturbations
       DqDx = DDXM.dot(fields)
       DqDz = DDZM.dot(fields)
       
       # Compute (partial) x derivatives of perturbations
       DZDXM = sps.diags(DZDX, offsets=0, format='csr')
       PqPx = DqDx - DZDXM_MT.dot(DqDz)
       
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
       WXZM = sps.diags(WXZ, offsets=0, format='csr')
       RdTM = sps.diags(RdT, offsets=0, format='csr')
       
       # Compute diagonal blocks related to sensible temperature
       RdT_bar = REFS[9]
       T_prime = (1.0 / Rd) * (RdT - RdT_bar[:,0])
       T_ratio = T_prime * np.reciprocal((1.0 / Rd) * RdT_bar[:,0])
       RdT_barM = sps.diags(RdT_bar[:,0], offsets=0, format='csr')
       PtPx = DDXM.dot(T_prime) - DZDX * DDZM.dot(T_prime)
       DtDz = DDZM.dot(T_prime)
       PtPxM = sps.diags(PtPx, offsets=0, format='csr')
       DtDzM = sps.diags(DtDz, offsets=0, format='csr')
       
       # Compute partial in X terrain following block
       PPXM = DDXM - DZDXM.dot(DDZM)
       PPXM_MT = DDXM - DZDXM_MT.dot(DDZM)
       
       # Compute common horizontal transport block
       UPXM = UM.dot(DDXM) + WXZM.dot(DDZM)
       
       bf = sps.diags(T_ratio + 1.0, offsets=0, format='csr')
       
       # Compute the blocks of the Jacobian operator
       LD11 = UPXM + PuPxM + DZDXM_BC.dot(DUDZM)
       LD12 = DuDzM + DUDZM
       LD13 = RdTM.dot(PPXM) + (Rd * PtPxM)
       LD14 = RdTM.dot(PlpPxM)
       
       LD21 = PwPxM + UBCM.dot(DDXM.dot(DZDXM_BC)) + WBCM.dot(PPXM + DZDXM.dot(DDZM))
       LD22 = UPXM + DwDzM
       LD23 = RdTM.dot(DDZM) + RdT_barM.dot(DLTDZM) + Rd * DtDzM
       LD24 = RdTM.dot(DlpDzM) - gc * bf
       
       LD31 = gam * PPXM_MT + PlpPxM + DZDXM_BC.dot(DLPDZM)
       LD32 = gam * DDZM + DlpDzM + DLPDZM
       LD33 = UPXM
       LD34 = None
       
       LD41 = PltPxM + DZDXM_BC.dot(DLPTDZM)
       LD42 = DltDzM + DLPTDZM
       LD43 = None
       LD44 = UPXM 
       
       # Null out Jacobian on dW at z = h(x)
       #LD12[np.ix_(botdex,botdex)] *= 0.0
       #LD22[np.ix_(botdex,botdex)] *= 0.0
       #LD32[np.ix_(botdex,botdex)] *= 0.0
       #LD42[np.ix_(botdex,botdex)] *= 0.0

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
       gam = PHYS[6]
       
       # Get the derivative operators
       DQDZ = REFG[4]
       DDXM = REFS[10]
       DDZM = REFS[11]
       DZDX = REFS[15]
       
       # Compute terrain following terms (two way assignment into fields)
       wxz = fields[:,1]
       WXZ = wxz - U * DZDX
       WXZ[botdex] *= 0.0
       
       # Compute advective (multiplicative) operators
       U = sps.diags(U, offsets=0, format='csr')
       wxz = sps.diags(wxz, offsets=0, format='csr')
       WXZ = sps.diags(WXZ, offsets=0, format='csr')
       
       # Get the static horizontal and vertical derivatives
       wDQDZ = wxz.dot(DQDZ)
       
       # Compute derivative of perturbations
       DqDx = DDXM.dot(fields)
       DqDz = DDZM.dot(fields)
       # Compute advection
       UDqDx = U.dot(DqDx)
       WDqDz = WXZ.dot(DqDz)
       transport = UDqDx + WDqDz + wDQDZ
       
       # Compute pressure gradient forces
       RdT_bar = REFS[9]
       RdT_prime = (RdT - RdT_bar[:,0])
       T_ratio = RdT_prime * np.reciprocal(RdT_bar[:,0])
       PGFX = RdT * (DqDx[:,2] - DZDX * DqDz[:,2])
       PGFZ = RdT * (DqDz[:,2]) - gc * T_ratio

       def DqDt():
              # Horizontal momentum equation
              DuDt = -(transport[:,0] + PGFX)
              # Vertical momentum equation
              DwDt = -(transport[:,1] + PGFZ)
              # Pressure (mass) equation
              LD33 = gam * (DqDx[:,0] - DZDX * DqDz[:,0] + DqDz[:,1])
              DpDt = -(transport[:,2] + LD33)
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
       
       # Null tendencies at essential vertical boundaries
       #DuDt[topdex] *= 0.0
       #DuDt[botdex] *= 0.0
       #DwDt[topdex] *= 0.0
       #DwDt[botdex] *= 0.0
       #DpDt[topdex] *= 0.0
       #DpDt[botdex] *= 0.0
       #DtDt[topdex] *= 0.0
       #DtDt[botdex] *= 0.0
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
       # Compute tendencies (2nd derivative term only)
       '''
       DuDt = RESCFX[udex] * DDx[:,0] + RESCFZ[udex] * DDz[:,0]
       DwDt = RESCFX[wdex] * DDx[:,1] + RESCFZ[wdex] * DDz[:,1]
       DpDt = RESCFX[pdex] * DDx[:,2] + RESCFZ[pdex] * DDz[:,2]
       DtDt = RESCFX[tdex] * DDx[:,3] + RESCFZ[tdex] * DDz[:,3]
       '''
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
       