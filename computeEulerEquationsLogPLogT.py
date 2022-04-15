#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import numpy as np
import warnings
import scipy.sparse as sps
import scipy.ndimage as spi
#import matplotlib.pyplot as plt

def enforceEssentialBC(sol, U, zeroDex, ebcDex, dhdx):
              
       # Enforce essential boundary conditions
       sol[zeroDex[0],0] = 0.0
       sol[zeroDex[1],1] = 0.0
       sol[zeroDex[2],2] = 0.0
       sol[zeroDex[3],3] = 0.0
       
       bdex = ebcDex[2]       
       sol[bdex,1] = dhdx * U[bdex]
       
       return sol

def enforceTendencyBC(DqDt, zeroDex, ebcDex, dhdx):
       
       DqDt[zeroDex[0],0] = 0.0
       DqDt[zeroDex[1],1] = 0.0
       DqDt[zeroDex[2],2] = 0.0
       DqDt[zeroDex[3],3] = 0.0
       
       bdex = ebcDex[2]  
       DqDt[bdex,1] = 0.0
       
       return DqDt

def computeRdT(q, RdT_bar, kap):
       
       # Compute pressure gradient force scaling (buoyancy)
       with warnings.catch_warnings():
              np.seterr(all='raise')
              
              earg = kap * q[:,2] + q[:,3]
              try:
                     #'''
                     T_ratio = np.expm1(earg)
                     RdT = RdT_bar * (T_ratio + 1.0)
                     #'''
                     '''
                     p_hat = np.exp(kap * fields[:,2], dtype=np.longdouble)
                     RdT_hat = p_hat * np.exp(fields[:,3], dtype=np.longdouble)
                     RdT = RdT_bar * RdT_hat
                     T_ratio = RdT_hat - 1.0
                     '''
              except FloatingPointError:
                     earg_max = np.amax(earg)
                     earg_min = np.amin(earg)
                     print('In argument to local T ratio: ', earg_min, earg_max)
                     pmax = np.amax(q[:,2])
                     pmin = np.amin(q[:,2])
                     print('Min/Max log pressures: ', pmin, pmax)
                     tmax = np.amax(q[:,3])
                     tmin = np.amin(q[:,3])
                     print('Min/Max log potential temperature: ', tmin, tmax)
                     # Compute buoyancy by approximation...
                     T_ratio = earg + 0.5 * np.power(earg, 2.0) # + ...
                     RdT_hat = 1.0 + T_ratio
                     RdT = RdT_bar * RdT_hat
                     
       return RdT, T_ratio

def computeFieldDerivatives(q, DDX, DDZ):
              
       DqDx = DDX.dot(q)
       DqDz = DDZ.dot(q)
       
       return DqDx, DqDz

def computeFieldDerivativeStag(q, DDX, DDZ):
       
       DqDx = DDX.dot(q)
       #DqDz = DDZ.dot(q)
       
       # This puts W with Ln_PT (Charney-Phillips style staggering)
       cdex = np.array([[0, 2, 1, 3]])
       qt = q[:,cdex]
       qs = np.reshape(qt, (2 * q.shape[0], 2), order='F')
       
       DqDz = DDZ.dot(qs)
       
       DqDz = np.reshape(DqDz, q.shape, order='F')
       DqDz = np.copy(DqDz[:,cdex])
         
       return DqDx[:,:], DqDz[:,0,:]

def computeFieldDerivatives2(PqPx, PqPz, DDX, DDZ, REFS, REFG, DCF, isFluxDiv):
          
       DQDZ = REFG[2]
       
       # Complete vertical parial
       PqPz += DQDZ
       
       if isFluxDiv:
              try:
                     vd = np.hstack((np.expand_dims(DCF[0], axis=1) * PqPx, \
                                     np.expand_dims(DCF[1], axis=1) * PqPz))
              except FloatingPointError:
                     vd = np.hstack((0.0 * PqPx, 0.0 * PqPz))
       else:
              vd = np.hstack((PqPx, PqPz))
       dvdx, dvdz = computeFieldDerivatives(vd, DDX, DDZ)
       
       P2qPx2 = dvdx[:,0:4]
       P2qPz2 = dvdz[:,4:] 
       
       P2qPzx = dvdz[:,0:4]
       P2qPxz = dvdx[:,4:]
       
       return P2qPx2, P2qPz2, P2qPzx, P2qPxz

def computePrepareFields(REFS, SOLT, INIT, udex, wdex, pdex, tdex):
       
       TQ = SOLT + INIT
       # Make the total quatities
       U = TQ[udex]
       W = TQ[wdex]
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')

       return fields, U, W

def computeRHS(fields, hydroState, DDX, DDZ, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, withRay):
       
       # Compute flow speed
       Q = fields + hydroState
       
       # Compute the updated RHS
       DqDx, DqDz = \
              computeFieldDerivatives(fields, DDX, DDZ)
       rhsVec = computeEulerEquationsLogPLogT_Explicit(PHYS, DqDx, DqDz, REFS, REFG, \
                                                     fields, Q[:,0], Q[:,1], ebcDex)
       if withRay:
              rhsVec += computeRayleighTendency(REFG, fields)
       
       rhsVec = enforceTendencyBC(rhsVec, zeroDex, ebcDex, dhdx)
       
       return rhsVec, DqDx, DqDz

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

# Fully explicit evaluation of the non linear equations (dynamic components)
def computeEulerEquationsLogPLogT_Explicit(PHYS, PqPx, DqDz, REFS, REFG, fields, U, W, ebcDex):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the Background fields
       RdT_bar = REFS[9][0]
       DQDZ = REFG[2]
              
       # Compute advective (multiplicative) operators
       UM = np.expand_dims(U,1)
       WM = np.expand_dims(W,1)
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute complete vertical partial
       PqPz = DqDz + DQDZ
       
       # Compute advection
       try:
              Uadvect = UM * PqPx
       except FloatingPointError:
              Uadvect = np.zeros(PqPx.shape)
       try:
              Wadvect = WM * PqPz
       except FloatingPointError:
              Wadvect = np.zeros(PqPz.shape)
       
       # Compute local divergence
       divergence = PqPx[:,0] + DqDz[:,1]
       
       divergence[ebcDex[0]] = PqPx[ebcDex[0],0]
       divergence[ebcDex[1]] = PqPx[ebcDex[1],0]
       
       # Compute pressure gradient forces
       pgradx = RdT * PqPx[:,2]
       pgradz = RdT * DqDz[:,2] - gc * T_ratio
       
       pgradx[ebcDex[0]] = 0.0
       pgradx[ebcDex[1]] = 0.0
       
       pgradz[ebcDex[0]] = 0.0
       pgradz[ebcDex[1]] = 0.0
       pgradz[ebcDex[2]] = 0.0
       pgradz[ebcDex[3]] = 0.0
       
       DqDt = -(Uadvect + Wadvect)
       
       # Horizontal momentum equation
       DqDt[:,0] -= pgradx
       # Vertical momentum equation
       DqDt[:,1] -= pgradz
       # Pressure (mass) equation
       try:
              DqDt[:,2] -= gam * divergence
       except FloatingPointError:
              DqDt[:,2] = np.zeros(PqPx.shape[0])
       # Potential Temperature equation (transport only)
       
       return DqDt

def computeRayleighTendency(REFG, fields):
       
       # Get the Rayleight operators
       mu = np.expand_dims(REFG[3],0)
       ROP = REFG[4]
       
       try:
              DqDt = -mu * ROP.dot(fields)
       except FloatingPointError:
              DqDt = np.zeros(fields.shape)
       
       #DqDt[:,2] = 0.0       
       
       return DqDt

def computeDiffusionTendency(q, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, REFS, REFG, ebcDex, DLD, DCF, isFluxDiv):
       
       dhdx = REFS[6][0]
       metrics = REFS[6][1]
       S = metrics[3]
       DQDZ = REFG[2]
       
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       DqDt = np.zeros(P2qPx2.shape)
       
       DC1 = DCF[0] # coefficient to the X direction flux
       DC2 = DCF[1] # coefficient to the Z direction flux
              
       try:
           #mu_xb = DC1[bdex]
           #mu_xt = DC1[tdex]
           
           #mu_xb = np.linalg.norm(np.stack((DC1[bdex],DC2[bdex]), axis=1), axis=1)
           #mu_xt = np.linalg.norm(np.stack((DC1[tdex],DC2[tdex]), axis=1), axis=1)
           
           mu_xb = np.reciprocal(S) * DC1[bdex]
           mu_xt = DC1[tdex]
       except FloatingPointError:
           mu_xb = np.zeros(bdex.shape)
           mu_xt = np.zeros(tdex.shape)
           
       if isFluxDiv:
              DC1 = 1.0
              DC2 = 1.0
              
       try:
              #%% INTERIOR DIFFUSION
              # Diffusion of u-w vector
              DqDt[:,0] = DC1 * (2.0 * P2qPx2[:,0]) + DC2 * (P2qPzx[:,1] + P2qPz2[:,0])
              DqDt[:,1] = DC1 * (P2qPx2[:,1] + P2qPxz[:,0]) + DC2 * (2.0 * P2qPz2[:,1])
              # Diffusion of scalars (broken up into anisotropic components
              DqDt[:,2] = DC1 * P2qPx2[:,2] + DC2 * P2qPz2[:,2]
              DqDt[:,3] = DC1 * P2qPx2[:,3] + DC2 * P2qPz2[:,3]
                  
              #'''        
              #%% TOP DIFFUSION (flow along top edge)
              DqDt[tdex,0] = mu_xt * P2qPx2[tdex,0]
              DqDt[tdex,1] = 0.0
              DqDt[tdex,2] = mu_xt * P2qPx2[tdex,2]
              DqDt[tdex,3] = mu_xt * P2qPx2[tdex,3]
              
              #%% BOTTOM DIFFUSION (flow along the terrain surface)
              
              # Compute directional derivatives
              DDX1 = REFS[16]
              DDX2 = REFS[17]
              SB = np.expand_dims(S, axis=1)
              DqDx = (DDX1 @ q[bdex,:])
              #DqDx = PqPx[bdex,:] + np.expand_dims(dhdx, axis=1) * PqPz[bdex,:]
              dqda = SB * (DqDx + np.expand_dims(dhdx, axis=1) * DQDZ[bdex,:])
              d2qda2 = SB * (DDX2 @ dqda)
       
              DqDt[bdex,0] = mu_xb * d2qda2[:,0]
              DqDt[bdex,1] = 0.0
              DqDt[bdex,2] = mu_xb * d2qda2[:,2]
              DqDt[bdex,3] = mu_xb * d2qda2[:,3]
              #'''
       except FloatingPointError:
              DqDt *= 0.0
       
       # Scale and apply coefficients
       DqDt[:,3] *= 0.71 / 0.4

       return DqDt