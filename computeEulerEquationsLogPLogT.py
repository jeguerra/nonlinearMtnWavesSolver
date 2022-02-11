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

def enforceTendencyBC(DqDt, zeroDex, ebcDex, dhdx):
       
       DqDt[zeroDex[0],0] = 0.0
       DqDt[zeroDex[1],1] = 0.0
       DqDt[zeroDex[2],2] = 0.0
       DqDt[zeroDex[3],3] = 0.0
       
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       DqDt[bdex,1] = 0.0 #np.copy(dhdx * DqDt[bdex,0])
       DqDt[tdex,1] = 0.0
       
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
       
       # This puts W with Ln_PT
       #'''
       qt = q[:,[0, 2, 1, 3]]
       qs = np.reshape(qt, (2 * q.shape[0], 2), order='F')
       DqDz = DDZ.dot(qs)
       DqDz = np.reshape(DqDz, q.shape, order='F')
       DqDz = np.copy(DqDz[:,[0, 2, 1, 3]])
       #'''
       '''
       # This keep the velocity vector on the same operator
       qs = np.reshape(q, (2 * q.shape[0], 2), order='F')
       DqDz = DDZ.dot(qs)
       DqDz = np.reshape(DqDz, q.shape, order='F')
       '''
       return DqDx, DqDz

def computeFieldDerivatives2(DqDx, DqDz, DDX, DDZ, REFS, REFG):
          
       DZDX = REFS[15]
       #DQDZ = REFG[2]
       #D2QDZ2 = REFG[-1]
       
       # Compute first partial in X (on CPU)
       PqPx = DqDx - DZDX * DqDz
       PqPz = DqDz# + DQDZ
       
       vd = np.hstack((PqPx, DqDz))
       dvdx, dvdz = computeFieldDerivatives(vd, DDX, DDZ)
       
       D2qDz2 = dvdz[:,4:] #DDZ.dot(DqDz)
       P2qPz2 = D2qDz2# + D2QDZ2
       D2qDx2 = dvdx[:,0:4] #DDX.dot(PqPx)
       
       P2qPzx = dvdz[:,0:4] #DDZ.dot(PqPx)
       P2qPxz = dvdx[:,4:] - DZDX * P2qPz2 #DDX.dot(DqDz)
       
       P2qPx2 = D2qDx2 - DZDX * P2qPzx
       
       return P2qPx2, P2qPz2, P2qPzx, P2qPxz, PqPx, PqPz

def computeFieldDerivativesFlux(DqDx, DqDz, DCF, DDX, DDZ, DZDX, DLD):
       
       # Compute first partial in X (on CPU)
       xFlux = DCF[0] * (DqDx - DZDX * DqDz)
       zFlux = DCF[1] * DqDz
       
       vd = np.hstack((xFlux, zFlux))
       DxFlux = DDX.dot(vd)
       DzFlux = DDZ.dot(vd)
       
       PxxFlux = (DxFlux[:,0:4] - DZDX * DzFlux[:,0:4])
       PxzFlux = DzFlux[:,0:4]
       PzzFlux = DzFlux[:,4:]
       PzxFlux = (DxFlux[:,4:] - DZDX * PzzFlux)
              
       return PxxFlux, PzzFlux, PxzFlux, PzxFlux, xFlux, zFlux

def computePrepareFields(REFS, SOLT, INIT, udex, wdex, pdex, tdex):
       
       TQ = SOLT + INIT
       # Make the total quatities
       U = TQ[udex]
       W = TQ[wdex]
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')

       return fields, U, W

def computeRHS(fields, hydroState, PHYS, REFS, REFG, ebcDex, zeroDex):
       
       # Compute flow speed
       Q = fields + hydroState
       UD = Q[:,0]
       WD = Q[:,1]
       
       # Compute the updated RHS
       DqDx, DqDz = \
              computeFieldDerivatives(fields, REFS[13][0], REFS[13][1])
       rhsVec = computeEulerEquationsLogPLogT_Explicit(PHYS, DqDx, DqDz, REFS, REFG, \
                                                     fields, UD, WD)
       #rhsVec += computeRayleighTendency(REFG, fields, zeroDex)
       
       rhsVec = enforceTendencyBC(rhsVec, zeroDex, ebcDex, REFS[6][0])
       
       return rhsVec, DqDx, DqDz

#%% Evaluate the Jacobian matrix
def computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, fields, U, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       Rd = PHYS[3]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the derivative operators (enhanced cubig spline derivative)
       DDXM = REFS[10][0]; DDZM = REFS[10][1]
       DZDX = REFS[15].flatten()
       
       DLTDZ = REFG[1]
       DQDZ = REFG[2]
       
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
def computeEulerEquationsLogPLogT_Explicit(PHYS, DqDx, DqDz, REFS, REFG, fields, U, W):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the Background fields
       RdT_bar = REFS[9][0]
       DZDX = REFS[15]
       DQDZ = REFG[2]
              
       # Compute advective (multiplicative) operators
       UM = np.expand_dims(U,1)
       WM = np.expand_dims(W,1)
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute partial and advection
       PqPx = DqDx - DZDX * DqDz
       PqPz = DqDz + DQDZ
       
       # Compute advection
       Uadvect = UM * PqPx
       Wadvect = WM * PqPz
       
       # Compute local divergence
       divergence = PqPx[:,0] + DqDz[:,1]
       
       # Compute pressure gradient forces
       pgradx = RdT * PqPx[:,2]
       pgradz = RdT * DqDz[:,2] - gc * T_ratio
       
       DqDt = -(Uadvect + Wadvect)
       
       # Horizontal momentum equation
       DqDt[:,0] -= pgradx
       # Vertical momentum equation
       DqDt[:,1] -= pgradz
       # Pressure (mass) equation
       DqDt[:,2] -= gam * divergence
       # Potential Temperature equation (transport only)
                        
       return DqDt

# Explicit advection RHS evaluation
def computeEulerEquationsLogPLogT_Advection(PHYS, DqDx, DqDz, REFS, REFG, fields, U, W):
       # Get physical constants
       kap = PHYS[4]       
       # Get the Background fields
       DQDZ = REFG[2]
       RdT_bar = REFS[9][0]
       DZDX = REFS[15]
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute partial and advection
       PqPx = DqDx - DZDX * DqDz
       PqPz = DqDz + DQDZ
       
       # Compute advection
       UM = np.expand_dims(U,1)
       WM = np.expand_dims(W,1)
       
       Uadvect = UM * PqPx
       Wadvect = WM * PqPz
       
       DqDt = -(Uadvect + Wadvect)
                        
       return DqDt

# Semi-implicit internal force evaluation
def computeEulerEquationsLogPLogT_InternalForce(PHYS, DqDx, DqDz, REFS, REFG, fields):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       # Get the Background fields
       DQDZ = REFG[2]
       RdT_bar = REFS[9][0]
       DZDX = REFS[15]
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute partial and advection
       PqPx = DqDx - DZDX * DqDz
       PqPz = DqDz + DQDZ
       
       # Compute local divergence
       divergence = PqPx[:,0] + PqPz[:,1]
       
       # Compute pressure gradient forces
       pgradx = RdT * PqPx[:,2]
       pgradz = RdT * DqDz[:,2] - gc * T_ratio
       
       DqDt = np.zeros(fields.shape)
       # Horizontal momentum equation
       DqDt[:,0] -= pgradx
       # Vertical momentum equation
       DqDt[:,1] -= pgradz
       # Pressure (mass) equation
       DqDt[:,2] -= gam * divergence
       # Potential Temperature equation (transport only)
                        
       return DqDt

def computeRayleighTendency(REFG, fields):
       
       # Get the Rayleight operators
       mu = np.expand_dims(REFG[3],0)
       ROP = REFG[4]
       
       DqDt = -mu * ROP.dot(fields)
              
       return DqDt

def computeDiffusionTendency(q, DqDx, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, REFS, REFG, ebcDex, zeroDex, DCF):
       
       DqDt = np.zeros(P2qPx2.shape)
       DC1 = DCF[0][:,0] # XX
       DC2 = DCF[1][:,0] # ZZ
            
       #%% INTERIOR DIFFUSION
       # Diffusion of u-w vector
       DqDt[:,0] = DC1 * (2.0 * P2qPx2[:,0]) + DC2 * (P2qPzx[:,1] + P2qPz2[:,0])  
       DqDt[:,1] = DC1 * (P2qPx2[:,1] + P2qPxz[:,0]) + DC2 * (2.0 * P2qPz2[:,1])
       # Diffusion of scalars (broken up into anisotropic components
       DqDt[:,2] = DC1 * P2qPx2[:,2] + DC2 * P2qPz2[:,2]
       DqDt[:,3] = DC1 * P2qPx2[:,3] + DC2 * P2qPz2[:,3]
                   
       #%% TOP DIFFUSION (flow along top edge)
       tdex = ebcDex[3]
       DqDt[tdex,0] = 2.0 * DC1[tdex] * P2qPx2[tdex,0]
       DqDt[tdex,1] = 0.0
       DqDt[tdex,2] = DC1[tdex] * P2qPx2[tdex,2]
       DqDt[tdex,3] = DC1[tdex] * P2qPx2[tdex,3]
       
       #%% BOTTOM DIFFUSION (flow along the terrain surface
       bdex = ebcDex[2]
       dhdx = REFS[6][0]
       dhx2 = np.power(dhdx, 2.0)
       
       metrics = REFS[6][1]
       d2hdx2 = metrics[0]
       dSdx = metrics[1]
       dSIdx = metrics[2]
       d2SIdx2 = metrics[3]
       S = metrics[4]
       S2 = metrics[5]
       #S4 = np.power(S2, 2.0)
       
       mu_x = DC1[bdex]
       mu_z = DC2[bdex]
       mu_t = S * np.linalg.norm(np.stack((mu_x, dhdx * mu_z)), axis=0)
       
       # Compute directional derivatives
       #DQDZ = REFG[2]
       qa = q[bdex,:]
       dqda = DqDx[bdex,:]# + np.expand_dims(dhdx, axis=1) * DQDZ[bdex,:]
       d2qda2 = P2qPx2[bdex,:] + np.expand_dims(dhx2, axis=1) * P2qPz2[bdex,:] + \
                2.0 * np.expand_dims(dhdx, axis=1) * P2qPxz[bdex,:] + \
                np.expand_dims(d2hdx2, axis=1) * PqPz[bdex,:]
       #'''       
       # dudt along terrain
       DqDt[bdex,0] = S * d2qda2[:,0] + S2 * dSIdx * dqda[:,0] + S * qa[:,0] * (dSdx * dSIdx + S * d2SIdx2)
       #DqDt[bdex,0] = S2 * d2qda2[:,0] + S * dSdx * dqda[:,0]
       DqDt[bdex,0] *= 2.0 * mu_t
       
       # dwdt along terrain
       DqDt[bdex,1] = 0.0
       
       # dlpdt along terrain
       DqDt[bdex,2] = S2 * d2qda2[:,2] + S * dSdx * dqda[:,2]
       DqDt[bdex,2] *= mu_t
       # dltdt along terrain
       DqDt[bdex,3] = S2 * d2qda2[:,3] + S * dSdx * dqda[:,3]
       DqDt[bdex,3] *= mu_t
       #'''
       
       # Scale and apply coefficients
       DqDt[:,3] *= 0.71 / 0.4

       return DqDt

def computeDiffusiveFluxTendency(DqDx, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, REFS, ebcDex, zeroDex, DCF):
       
       DZDX = REFS[15]
       DDXM = REFS[16]
       DqDt = np.zeros(P2qPx2.shape)
            
       #%% INTERIOR DIFFUSION
       # Diffusion of u-w vector
       DqDt[:,0] = (2.0 * P2qPx2[:,0] + P2qPzx[:,1] + P2qPz2[:,0])  
       DqDt[:,1] = (P2qPx2[:,1] + P2qPxz[:,0] + 2.0 * P2qPz2[:,1])
       # Diffusion of scalars (broken up into anisotropic components
       DqDt[:,2] = (P2qPx2[:,2] + P2qPz2[:,2])
       DqDt[:,3] = (P2qPx2[:,3] + P2qPz2[:,3])
       #'''            
       #%% TOP DIFFUSION (flow along top edge)
       tdex = ebcDex[3]
       DqDt[tdex,0] = 2.0 * P2qPx2[tdex,0]
       DqDt[tdex,2] = P2qPx2[tdex,2]
       DqDt[tdex,3] = P2qPx2[tdex,3]
       
       #%% BOTTOM DIFFUSION (flow along the terrain surface)
       bdex = ebcDex[2]
       dhx = DZDX[bdex,0]
       dhx2 = np.power(dhx, 2.0)
       S2 = np.reciprocal(1.0 + dhx2)
       
       mu_x = DCF[0][bdex,0]
       mu_z = DCF[1][bdex,0]
       dqb = DqDx[bdex,:]
       
       # Compute TF fluxes
       flux1_u = S2 * (dqb[:,0] + 0.5 * dqb[:,1])
       flux2_u = 0.5 * S2 * dhx * dqb[:,0]
       
       flux1_w = 0.5 * S2 * dqb[:,1]
       flux2_w = S2 * dhx * (dqb[:,1] + 0.5 * dqb[:,0])
       
       flux1_p = S2 * dqb[:,2]
       flux2_p = S2 * dhx * dqb[:,2]
       
       flux1_t = S2 * dqb[:,3]
       flux2_t = S2 * dhx * dqb[:,3]
       
       # Compute flux gradients
       fluxes = np.stack((mu_x * flux1_u, mu_z * flux2_u, mu_x * flux1_w, mu_z * flux2_w, \
                          mu_x * flux1_p, mu_z * flux2_p, mu_x * flux1_t, mu_z * flux2_t), axis=1)
       dflx = DDXM @ (fluxes)
              
       # dudt along terrain
       DqDt[bdex,0] = 2.0 * S2 * (mu_x * dflx[:,0] + dhx * mu_z * dflx[:,1])
       # dwdt along terrain
       DqDt[bdex,1] = 0.0
       
       # dlpdt along terrain
       DqDt[bdex,2] = S2 * (mu_x * dflx[:,4] + dhx * mu_z * dflx[:,5])
       # dltdt along terrain
       DqDt[bdex,3] = S2 * (mu_x * dflx[:,6] + dhx * mu_z * dflx[:,7])
       
       # Scale and apply coefficients
       Pr = 0.71 / 0.4
       DqDt[:,3] *= 1.0*Pr
              
       return DqDt