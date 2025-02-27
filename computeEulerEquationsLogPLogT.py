#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import math as mt
import torch as tch
import numpy as np
import bottleneck as bn
import scipy.sparse as sps
# Change floating point errors
np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')

def enforceBC_RHS(rhs, ebcDex):
       
       ldex = ebcDex[0]
       rdex = ebcDex[1]
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       # INFLOW LEFT      
       rhs[ldex,:] = 0.0
       
       # OUTFLOW RIGHT
       rhs[rdex,0] = 0.0 
       rhs[rdex,1] = 0.0 
       rhs[rdex,2] = 0.0
       rhs[rdex,3] = 0.0
       
       # TERRAIN
       rhs[bdex,0:2] = 0.0
       rhs[bdex,3] = 0.0
       
       # MODEL TOP
       rhs[tdex,1] = 0.0
       
       return rhs

def enforceBC_SOL(sol, ebcDex, init):
       
       ldex = ebcDex[0]
       rdex = ebcDex[1]
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       # INFLOW LEFT      
       sol[ldex,:] = init[ldex,:]
       
       # OUTFLOW RIGHT
       sol[rdex,0] = 0.0 #init[rdex,0]
       sol[rdex,1] = 0.0 #init[rdex,1]
       sol[rdex,2] = init[rdex,2]
       sol[rdex,3] = init[rdex,3]
       
       # TERRAIN
       sol[bdex,0:2] = 0.0
       sol[bdex,3] = init[bdex,3]
       
       # MODEL TOP
       sol[tdex,1] = 0.0
       
       return sol

def computeNewTimeStep(PHYS, RdT, fields, DLD, isInitial=False):
       
       # Compute new time step based on maximum local sound speed
       if tch.is_tensor(RdT):
              VWAV_max = (PHYS[6] * RdT).sqrt().max()
       else:
              VWAV_max = np.sqrt(PHYS[6] * RdT).max()
                            
       # Perform some checks before setting the new DT
       try:
              DT = DLD[4] / VWAV_max
       except:
              DT *= 0.5
      
       if isInitial:
              # Bisect time step on initialization
              DT *= 0.1
              
       return DT, VWAV_max

def computeRdT(PHYS, sol, pert, RdT_bar):
       
       kap = PHYS[4]
       
       # Compute pressure gradient force scaling (buoyancy)              
       earg = kap * pert[:,2] + pert[:,3]
       if tch.is_tensor(earg):
              T_ratio = tch.expm1(earg)
       else:
              T_ratio = np.expm1(earg)
       RdT = RdT_bar * (T_ratio + 1.0)                 
                     
       return RdT, T_ratio

def computeFieldDerivative(q, DD, RSBops):
                            
       if RSBops:
              Dq = DD.dot(q)
       else:
              Dq = tch.matmul(DD,
                                tch.from_numpy(q)).numpy()
              
       return Dq

def computePrepareFields(OPS, SOLT, udex, wdex, pdex, tdex):
       
       # Make the total quatities
       U = SOLT[udex]
       W = SOLT[wdex]
       
       fields = np.reshape(SOLT, (OPS, 4), order='F')

       return fields, U, W

def computeRHS(state, fields, DDX, DDZ, PHYS, REFS, REFG, withRay, isTFOpX):
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(PHYS, state, fields, REFS[9][0])
       
       # Compute the updated RHS
       DqDx = DDX @ fields
       DqDz = DDZ @ fields
              
       if isTFOpX:
              PqPx = np.copy(DqDx)
       else:
              PqPx = DqDx - REFS[14] * DqDz
       
       PqPz = DqDz + REFG[3]
       rhsVec = computeEulerEquationsLogPLogT_Explicit(PHYS, PqPx, PqPz, DqDz, 
                                                       RdT, T_ratio, state)
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
       DZDX = REFS[14].flatten()
       
       DZDXM = sps.diags(DZDX, offsets=0, format='csr')
       PPXM = DDXM - DZDXM.dot(DDZM)
       
       DLTDZ = REFG[2]
       DQDZ = REFG[3]
       
       # Compute terrain following terms (two way assignment into fields)
       wxz = np.array(fields[:,1])
       UZX = U * DZDX
       WXZ = wxz - UZX

       # Compute raw derivatives of perturbations
       DqDx = DDXM @ fields
       DqDz = DDZM @ fields
       
       # Compute terrain following x derivatives of perturbations
       DZDXM = sps.diags_array(DZDX, offsets=0, format='csr')
       PqPx = DqDx - DZDXM.dot(DqDz)
       
       # Compute partial in X terrain following block
       PPXM = DDXM - DZDXM.dot(DDZM)
       
       # Compute vertical gradient diagonal operators
       DuDzM = sps.diags_array(DqDz[:,0], offsets=0, format='csr')
       DwDzM = sps.diags_array(DqDz[:,1], offsets=0, format='csr')
       DlpDzM = sps.diags_array(DqDz[:,2], offsets=0, format='csr')
       DltDzM = sps.diags_array(DqDz[:,3], offsets=0, format='csr')
       
       # Compute horizontal gradient diagonal operators
       PuPxM = sps.diags_array(PqPx[:,0], offsets=0, format='csr')
       PwPxM = sps.diags_array(PqPx[:,1], offsets=0, format='csr')
       PlpPxM = sps.diags_array(PqPx[:,2], offsets=0, format='csr')
       PltPxM = sps.diags_array(PqPx[:,3], offsets=0, format='csr')
       
       # Compute hydrostatic state diagonal operators
       DLTDZM = sps.diags_array(DLTDZ[:,0], offsets=0, format='csr')
       DUDZM = sps.diags_array(DQDZ[:,0], offsets=0, format='csr')
       DLPDZM = sps.diags_array(DQDZ[:,2], offsets=0, format='csr')
       DLPTDZM = sps.diags_array(DQDZ[:,3], offsets=0, format='csr')
       
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
       
       PtPxM = sps.diags_array(PtPx, offsets=0, format='csr')
       DtDzM = sps.diags_array(DtDz, offsets=0, format='csr')
       
       # Compute advective (multiplicative) diagonal operators
       UM = sps.diags_array(U, offsets=0, format='csr')
       WXZM = sps.diags_array(WXZ, offsets=0, format='csr')
       
       # Compute common horizontal transport block
       UPXM = UM @ DDXM + WXZM @ DDZM
       
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
       UM = sps.diags_array(UZ, offsets=0, format='csr')
       PORZM = sps.diags_array(PORZ, offsets=0, format='csr')
       
       # Compute hydrostatic state diagonal operators
       DLTDZ = REFG[2]
       DQDZ = REFG[3]
       DLTDZM = sps.diags_array(DLTDZ[:,0], offsets=0, format='csr')
       DUDZM = sps.diags_array(DQDZ[:,0], offsets=0, format='csr')
       DLPDZM = sps.diags_array(DQDZ[:,2], offsets=0, format='csr')
       DLPTDZM = sps.diags_array(DQDZ[:,3], offsets=0, format='csr')
       unit = sps.identity(OPS)
              
       #%% Compute the terms in the equations
       U0DDX = UM @ DDXM
       
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
def computeAdvectionLogPLogT_Explicit(PHYS, PqPx, PqPz, state):
       
       # Compute advective (multiplicative) operators
       if tch.is_tensor(state):
              UM = tch.unsqueeze(state[:,0], dim=1)
              WM = tch.unsqueeze(state[:,1], dim=1)
       else:
              UM = np.expand_dims(state[:,0], axis=1)
              WM = np.expand_dims(state[:,1], axis=1)
       
       # Compute advection
       DqDt = -(UM * PqPx + WM * PqPz)
       
       return DqDt

# Fully explicit evaluation of the non linear internal forcing
def computeInternalForceLogPLogT_Explicit(PHYS, PqPx, PqPz, RdT, T_ratio, DqDt):
       # Get physical constants
       gc = PHYS[0]
       gam = PHYS[6]
              
       # Horizontal momentum equation
       DqDt[:,0] -= RdT * PqPx[:,2]
       # Vertical momentum equation
       #DqDt[:,1] -= (RdT * PqPz[:,2] - gc * T_ratio)
       DqDt[:,1] -= (RdT * PqPz[:,2] + gc)
       # Divergence
       DqDt[:,2] -= gam * (PqPx[:,0] + PqPz[:,1])
       # Potential temperature equation (material derivative)
       
       return DqDt

# Fully explicit evaluation of the non linear equations (dynamic components)
def computeEulerEquationsLogPLogT_Explicit(PHYS, PqPx, PqPz, DqDz, 
                                           RdT, T_ratio, state):
       
       DqDt = computeAdvectionLogPLogT_Explicit(PHYS, PqPx, PqPz, state)
       
       DqDt = computeInternalForceLogPLogT_Explicit(PHYS, PqPx, PqPz, RdT, T_ratio, DqDt)
       
       return DqDt

def computeRayleighTendency(REFG, fields):
       
       # Get the Rayleight operators
       ROP = REFG[0][-1]
       DqDt = -ROP * fields
              
       return DqDt

#@njit(parallel=True)
def computeDiffusionTendency(P2qPx2, P2qPz2, P2qPzx, P2qPxz, ebcDex):
       
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       if tch.is_tensor(P2qPx2):
              DqDt = 0.0 * P2qPx2.clone()
       else:
              DqDt = 0.0 * np.copy(P2qPx2)
       
       #%% INTERIOR DIFFUSION
       # Diffusion of u-w vector
       DqDt[:,0] = 0.5 * ((2.0 * P2qPx2[:,0]) + (P2qPxz[:,1] + P2qPz2[:,0]))
       DqDt[:,1] = 0.5 * ((P2qPx2[:,1] + P2qPzx[:,0]) + (2.0 * P2qPz2[:,1]))
       # Diffusion of scalars (broken up into anisotropic components
       DqDt[:,2] = P2qPx2[:,2] + P2qPz2[:,2]
       DqDt[:,3] = P2qPx2[:,3] + P2qPz2[:,3]
       #'''   
       #%% TOP DIFFUSION (flow along top edge)
       DqDt[tdex,:] = P2qPx2[tdex,:]
       
       #%% BOTTOM DIFFUSION (flow along the terrain surface)
       DqDt[bdex,:] = P2qPx2[bdex,:]

       return DqDt
