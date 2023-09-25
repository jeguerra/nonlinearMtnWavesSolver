#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import numpy as np
import scipy.sparse as sps
import sparse_dot_mkl as spk
from numba import njit
# Change floating point errors
np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')

def enforceBC_RHS(rhs, ebcDex):
       
       ldex = ebcDex[0]
       rdex = ebcDex[1]
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       vdex = np.concatenate((bdex, tdex))
       
       # Tendencies consistent with field conditions
       rhs[ldex,:] = 0.0
       rhs[vdex,0] = 0.0
       rhs[vdex,1] = 0.0
       
       return rhs

def computeRdT(q, RdT_bar, kap):
       
       # Compute pressure gradient force scaling (buoyancy)              
       earg = kap * q[:,2] + q[:,3]
       T_ratio = np.expm1(earg)#, dtype=np.float64)
       #T_exp = np.exp(earg, dtype=np.longdouble)                 
              
       RdT = RdT_bar * (T_ratio + 1.0)
       #RdT = RdT_bar * T_exp
       #T_ratio = T_exp - 1.0
                     
       #return RdT.astype(np.float64), T_ratio.astype(np.float64)
       return RdT, T_ratio

def computeFieldDerivatives(q, DDX, DDZ, verticalStagger, RSBops):
                     
       if verticalStagger:
              qs = np.reshape(q, (4 * q.shape[0], 1), order='F')
              
              if RSBops:
                     DqDx = DDX.dot(q)
                     DqDz = DDZ.dot(qs)
              else:
                     DqDx = spk.dot_product_mkl(DDX, q)
                     DqDz = spk.dot_product_mkl(DDZ, qs)
              
              #DqDx = np.reshape(DqDx, q.shape, order='F')
              DqDz = np.reshape(DqDz, q.shape, order='F')
       else:
              if RSBops:
                     DqDx = DDX.dot(q)
                     DqDz = DDZ.dot(q)
              else:
                     DqDx = spk.dot_product_mkl(DDX, q)
                     DqDz = spk.dot_product_mkl(DDZ, q)
              
       return DqDx, DqDz

def computeFieldDerivatives2(PqPx, PqPz, DDX, DDZ, REFS, RSBops):
       
       vd = np.hstack((PqPx, PqPz))
       pvpx, dvdz = computeFieldDerivatives(vd, DDX, DDZ, False, RSBops)
       
       P2qPx2 = pvpx[:,0:4]
       P2qPz2 = dvdz[:,4:] 
       
       P2qPzx = dvdz[:,0:4]
       P2qPxz = pvpx[:,4:]
       
       return P2qPx2, P2qPz2, P2qPzx, P2qPxz

def computePrepareFields(REFS, SOLT, INIT, udex, wdex, pdex, tdex):
       
       #TQ = SOLT + INIT
       # Make the total quatities
       U = SOLT[udex]
       W = SOLT[wdex]
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')

       return fields, U, W

def computeRHS(fields, hydroState, DDX, DDZ, dhdx, PHYS, REFS, REFG, withRay, vertStagger, isTFOpX, RSBops):
       
       # Compute flow speed
       Q = np.copy(fields)
       Q[:,2:] += hydroState[:,2:]
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, REFS[9][0], PHYS[4])
       
       # Compute the updated RHS
       PqPx, DqDz = computeFieldDerivatives(fields, DDX, DDZ, vertStagger, RSBops)
              
       if not isTFOpX:
              PqPx -= REFS[15] * DqDz
                            
       rhsVec = computeEulerEquationsLogPLogT_Explicit(PHYS, PqPx, DqDz, REFG[2], RdT, T_ratio, \
                                                       fields, Q)
       if withRay:
              rhsVec += computeRayleighTendency(REFG, fields)
       
       return rhsVec, PqPx, DqDz

#%% Evaluate the Jacobian matrix
def computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, fields, U, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       Rd = PHYS[3]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the derivative operators (enhanced cubig spline derivative)
       DDXM = REFS[10][0]
       DDZM = REFS[10][1]
       DZDX = REFS[15].flatten()
       
       DZDXM = sps.diags(DZDX, offsets=0, format='csr')
       PPXM = DDXM - DZDXM.dot(DDZM)
       
       DLTDZ = REFG[1]
       DQDZ = REFG[2]
       
       # Compute terrain following terms (two way assignment into fields)
       wxz = np.array(fields[:,1])
       UZX = U * DZDX
       WXZ = wxz - UZX

       # Compute raw derivatives of perturbations
       DqDx = DDXM.dot(fields)
       DqDz = DDZM.dot(fields)
       
       # Compute terrain following x derivatives of perturbations
       DZDXM = sps.diags(DZDX, offsets=0, format='csr')
       PqPx = DqDx - DZDXM.dot(DqDz)
       
       # Compute partial in X terrain following block
       PPXM = DDXM - DZDXM.dot(DDZM)
       
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
       DLTDZM = sps.diags(DLTDZ[:,0], offsets=0, format='csr')
       DUDZM = sps.diags(DQDZ[:,0], offsets=0, format='csr')
       DLPDZM = sps.diags(DQDZ[:,2], offsets=0, format='csr')
       DLPTDZM = sps.diags(DQDZ[:,3], offsets=0, format='csr')
       
       # Compute diagonal blocks related to sensible temperature
       RdT_bar = REFS[9][0]
       T_bar = (1.0 / Rd) * RdT_bar
       
       targ = kap * fields[:,2] + fields[:,3]
       T_ratio = np.expm1(targ) 
       bf = T_ratio + 1.0
       RdT = RdT_bar * bf
       
       # Compute T'
       T_prime = T_ratio * T_bar
       
       RdT_barM = sps.diags(RdT_bar, offsets=0, format='csr')
       RdTM = sps.diags(RdT, offsets=0, format='csr')
       bfM = sps.diags(bf, offsets=0, format='csr')
       
       # Compute derivatives of temperature perturbation
       PtPx = PPXM.dot(T_prime)
       DtDz = DDZM.dot(T_prime)
       
       PtPxM = sps.diags(PtPx, offsets=0, format='csr')
       DtDzM = sps.diags(DtDz, offsets=0, format='csr')
       
       # Compute advective (multiplicative) diagonal operators
       UM = sps.diags(U, offsets=0, format='csr')
       WXZM = sps.diags(WXZ, offsets=0, format='csr')
       
       # Compute common horizontal transport block
       UPXM = UM.dot(DDXM) + WXZM.dot(DDZM)
       
       # Compute the blocks of the Jacobian operator
       LD11 = UPXM + PuPxM
       LD12 = DuDzM + DUDZM
       LD13 = RdTM.dot(PPXM) + (Rd * PtPxM)
       LD14 = RdTM.dot(PlpPxM)
       
       LD21 = PwPxM
       LD22 = UPXM + DwDzM
       LD23 = RdTM.dot(DDZM) + RdT_barM.dot(DLTDZM) + Rd * DtDzM
       LD24 = RdTM.dot(DlpDzM) - gc * bfM
       
       LD31 = gam * PPXM + PlpPxM
       LD32 = gam * DDZM + DlpDzM + DLPDZM
       LD33 = UPXM
       LD34 = None
       
       LD41 = PltPxM
       LD42 = DltDzM + DLPTDZM
       LD43 = None
       LD44 = UPXM
       
       DOPS = [LD11, LD12, LD13, LD14, \
               LD21, LD22, LD23, LD24, \
               LD31, LD32, LD33, LD34, \
               LD41, LD42, LD43, LD44]
       
       return DOPS
    
#%% The linear equation operator
def computeEulerEquationsLogPLogT_Classical(DIMS, PHYS, REFS, REFG):
       # Get physical constants
       gc = PHYS[0]
       gam = PHYS[6]
       
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = NX * NZ
       
       # Get REFS data
       UZ = REFS[8]
       PORZ = REFS[9][0]
       # Full spectral transform derivative matrices
       DDXM = REFS[10][0]
       DDZM = REFS[10][1]
              
       #%% Compute the various blocks needed
       UM = sps.diags(UZ, offsets=0, format='csr')
       PORZM = sps.diags(PORZ, offsets=0, format='csr')
       
       # Compute hydrostatic state diagonal operators
       DLTDZ = REFG[1]
       DQDZ = REFG[2]
       DLTDZM = sps.diags(DLTDZ[:,0], offsets=0, format='csr')
       DUDZM = sps.diags(DQDZ[:,0], offsets=0, format='csr')
       DLPDZM = sps.diags(DQDZ[:,2], offsets=0, format='csr')
       DLPTDZM = sps.diags(DQDZ[:,3], offsets=0, format='csr')
       unit = sps.identity(OPS)
              
       #%% Compute the terms in the equations
       U0DDX = UM.dot(DDXM)
       
       # Horizontal momentum
       LD11 = U0DDX
       LD12 = DUDZM
       LD13 = PORZM.dot(DDXM)
       LD14 = sps.csr_matrix((OPS,OPS))
       
       # Vertical momentum
       LD21 = sps.csr_matrix((OPS,OPS))
       LD22 = U0DDX
       LD23 = PORZM.dot(DDZM + DLTDZM)
       # Equivalent form from direct linearization
       #LD23 = PORZM.dot(DDZM) + gc * (1.0 / gam - 1.0) * unit
       LD24 = -gc * unit
       
       # Log-P equation
       LD31 = gam * DDXM
       LD32 = gam * DDZM + DLPDZM
       LD33 = U0DDX
       LD34 = None
       
       # Log-Theta equation
       LD41 = sps.csr_matrix((OPS,OPS))
       LD42 = DLPTDZM
       LD43 = None
       LD44 = U0DDX
       
       DOPS = [LD11, LD12, LD13, LD14, \
               LD21, LD22, LD23, LD24, \
               LD31, LD32, LD33, LD34, \
               LD41, LD42, LD43, LD44]
       
       return DOPS

# Fully explicit evaluation of the non linear advection
def computeAdvectionLogPLogT_Explicit(PHYS, PqPx, PqPz, fields, state):
       
       # Compute advective (multiplicative) operators
       UM = np.expand_dims(state[:,0], axis=1)
       WM = np.expand_dims(state[:,1], axis=1)
       
       # Compute advection
       DqDt = -(UM * PqPx + WM * PqPz)
       
       return DqDt

# Fully explicit evaluation of the non linear internal forcing
def computeInternalForceLogPLogT_Explicit(PHYS, PqPx, DqDz, RdT, T_ratio, DqDt):
       # Get physical constants
       gc = PHYS[0]
       gam = PHYS[6]
              
       # Horizontal momentum equation
       DqDt[:,0] -= RdT * PqPx[:,2]
       # Vertical momentum equation
       DqDt[:,1] -= (RdT * DqDz[:,2] - gc * T_ratio)
       # Pressure (mass) equation
       DqDt[:,2] -= gam * (PqPx[:,0] + DqDz[:,1])
       # Potential temperature equation (material derivative)
       
       return DqDt

# Fully explicit evaluation of the non linear equations (dynamic components)
def computeEulerEquationsLogPLogT_Explicit(PHYS, PqPx, DqDz, DQDZ, RdT, T_ratio, fields, state):

       # Compute complete vertical partial
       PqPz = np.copy(DqDz)
       PqPz[:,2:] += DQDZ[:,2:]
       
       DqDt = computeAdvectionLogPLogT_Explicit(PHYS, PqPx, PqPz, fields, state)
       
       DqDt = computeInternalForceLogPLogT_Explicit(PHYS, PqPx, DqDz, RdT, T_ratio, DqDt)
       
       return DqDt

def computeRayleighTendency(REFG, fields):
       
       # Get the Rayleight operators
       mu = np.expand_dims(REFG[3],0)
       ROP = REFG[4][0]
       rdex = REFG[-1]
       
       DqDt = np.zeros(fields.shape)
       try:
              DqDt[:,rdex] = -mu * ROP.dot(fields[:,rdex])
       except FloatingPointError:
              DqDt[:,rdex] = 0.0
              
       return DqDt

#@njit(parallel=True)
def computeDiffusionTendency(P2qPx2, P2qPz2, P2qPzx, P2qPxz, ebcDex, DLD):
       
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       DqDt = np.zeros(P2qPx2.shape)
       
       #%% INTERIOR DIFFUSION
       # Diffusion of u-w vector
       DqDt[:,0] = (2.0 * P2qPx2[:,0]) + (P2qPzx[:,1] + P2qPz2[:,0])
       DqDt[:,1] = (P2qPx2[:,1] + P2qPxz[:,0]) + (2.0 * P2qPz2[:,1])
       # Diffusion of scalars (broken up into anisotropic components
       DqDt[:,2] = P2qPx2[:,2] + P2qPz2[:,2]
       DqDt[:,3] = P2qPx2[:,3] + P2qPz2[:,3]
       #'''   
       #%% TOP DIFFUSION (flow along top edge)
       DqDt[tdex,:] = P2qPx2[tdex,:]
       
       #%% BOTTOM DIFFUSION (flow along the terrain surface)
       DqDt[bdex,:] = P2qPx2[bdex,:]

       return DqDt