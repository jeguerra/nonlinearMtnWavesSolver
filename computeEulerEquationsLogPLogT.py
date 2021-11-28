#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import numpy as np
import warnings
import scipy.sparse as sps
import matplotlib.pyplot as plt

def enforceTerrainTendency(DqDt, ebcDex, DZDX):
       
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       DqDt[bdex,1] = np.copy(DZDX[bdex,0] * DqDt[bdex,0])
       DqDt[tdex,1] = 0.0
       
       return DqDt

def enforceTendencyBC(DqDt, zeroDex):
       
       DqDt[zeroDex[0],0] = 0.0
       DqDt[zeroDex[1],1] = 0.0
       DqDt[zeroDex[2],2] = 0.0
       DqDt[zeroDex[3],3] = 0.0
       
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
       
       #print(DDX.shape)
       #print(DDZ.shape)
       
       DqDx = DDX.dot(q)
       DqDz = DDZ.dot(q)
       
       return DqDx, DqDz

def computeFieldDerivatives2(DqDx, DqDz, REFG, DDX, DDZ, DZDX):
       
       DQDZ = REFG[2]
       D2QDZ2 = REFG[-1]
       
       # Compute first partial in X (on CPU)
       PqPx = DqDx - DZDX * DqDz
       PqPz = DqDz# + DQDZ
       
       vd = np.hstack((PqPx, DqDz))
       dvdx, dvdz = computeFieldDerivatives(vd, DDX, DDZ)
       
       D2qDz2 = dvdz[:,4:] #DDZ.dot(DqDz)
       P2qPz2 = D2qDz2 + D2QDZ2
       D2qDx2 = dvdx[:,0:4] #DDX.dot(PqPx)
       
       P2qPzx = dvdz[:,0:4] #DDZ.dot(PqPx)
       P2qPxz = dvdx[:,4:] - DZDX * P2qPz2 #DDX.dot(DqDz)
       
       P2qPx2 = D2qDx2 - DZDX * P2qPzx
       
       return P2qPx2, P2qPz2, P2qPzx, P2qPxz, PqPx, PqPz

def computeFieldDerivativesFlux(DqDx, DqDz, DCF, REFG, DDX, DDZ, DZDX):
       
       DQDZ = REFG[2]
       #D2QDZ2 = REFG[-1]
       
       # Compute first partial in X (on CPU)
       xFlux = DCF[0] * (DqDx - DZDX * DqDz)
       zFlux = DCF[1] * DqDz# + DQDZ)
       
       vd = np.hstack((xFlux, zFlux))
       DxFlux = DDX.dot(vd)
       DzFlux = DDZ.dot(vd)
       
       PxxFlux = DxFlux[:,0:4] - DZDX * DzFlux[:,0:4]
       PxzFlux = DzFlux[:,0:4]
       PzzFlux = DzFlux[:,4:]
       PzxFlux = DxFlux[:,4:] - DZDX * PzzFlux
              
       return PxxFlux, PzzFlux, PxzFlux, PzxFlux, xFlux, zFlux

def computePrepareFields(REFS, SOLT, INIT, udex, wdex, pdex, tdex):
       
       TQ = SOLT + INIT
       # Make the total quatities
       U = TQ[udex]
       W = TQ[wdex]
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')

       return fields, U, W

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
       NZ = DIMS[4]
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

# Fully explicit evaluation of the non linear equations
def computeEulerEquationsLogPLogT_StaticResidual(PHYS, DqDx, DqDz, REFG, DZDX, RdT_bar, fields, U, W, ebcDex, zeroDex):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the Background fields
       DQDZ = REFG[2]
              
       # Compute advective (multiplicative) operators
       UM = np.expand_dims(U,1)
       WM = np.expand_dims(W,1)
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute partial and advection
       PqPx = DqDx - DZDX * DqDz
       
       # Compute advection
       Uadvect = UM * PqPx
       Wadvect = WM * (DqDz + DQDZ)
       
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
       
       DqDt = enforceTendencyBC(DqDt, zeroDex)
       DqDt = enforceTerrainTendency(DqDt, ebcDex, DZDX)
                               
       return DqDt

# Fully explicit evaluation of the non linear equations (dynamic components)
def computeEulerEquationsLogPLogT_Explicit(PHYS, DqDx, DqDz, REFG, DZDX, RdT_bar, fields, U, W, ebcDex, zeroDex):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the Background fields
       DQDZ = REFG[2]
              
       # Compute advective (multiplicative) operators
       UM = np.expand_dims(U,1)
       WM = np.expand_dims(W,1)
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute partial and advection
       PqPx = DqDx - DZDX * DqDz
       
       # Compute advection
       Uadvect = UM * PqPx
       Wadvect = WM * (DqDz + DQDZ)
       
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
       
       DqDt = enforceTendencyBC(DqDt, zeroDex)
       DqDt = enforceTerrainTendency(DqDt, ebcDex, DZDX)
                        
       return DqDt

# Explicit advection RHS evaluation
def computeEulerEquationsLogPLogT_Advection(PHYS, DqDx, DqDz, REFG, DZDX, RdT_bar, fields, U, W, ebcDex, zeroDex):
       # Get physical constants
       kap = PHYS[4]       
       # Get the Background fields
       DQDZ = REFG[2]
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute partial and advection
       PqPx = DqDx - DZDX * DqDz
       
       # Compute advection
       UM = np.expand_dims(U,1)
       WM = np.expand_dims(W,1)
       
       Uadvect = UM * PqPx
       Wadvect = WM * (DqDz + DQDZ)
       
       DqDt = -(Uadvect + Wadvect)
       
       # Compute local divergence
       #gam = PHYS[6]
       #divergence = PqPx[:,0] + DqDz[:,1]
       #DqDt[:,2] -= gam * divergence
       
       DqDt = enforceTendencyBC(DqDt, zeroDex)
       DqDt = enforceTerrainTendency(DqDt, ebcDex, DZDX)
                        
       return DqDt

# Semi-implicit internal force evaluation
def computeEulerEquationsLogPLogT_InternalForce(PHYS, DqDt_adv, DqDx, DqDz, REFG, DZDX, RdT_bar, fields, ebcDex, zeroDex):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Compute pressure gradient force scaling (buoyancy)
       RdT, T_ratio = computeRdT(fields, RdT_bar, kap)
       
       # Compute partial and advection
       PqPx = DqDx - DZDX * DqDz
       
       # Compute local divergence
       divergence = PqPx[:,0] + DqDz[:,1]
       
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
       
       DqDt = enforceTendencyBC(DqDt, zeroDex)
       DqDt = enforceTerrainTendency(DqDt, ebcDex, DZDX)
                        
       return DqDt

def computeRayleighTendency(REFG, fields, zeroDex):
       
       # Get the Rayleight operators
       mu = np.expand_dims(REFG[3],0)
       ROP = REFG[4]
       
       DqDt = -mu * ROP.dot(fields)
       
       DqDt = enforceTendencyBC(DqDt, zeroDex)
       
       return DqDt

def computeDiffusionTendency(PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, REFS, ebcDex, zeroDex, DCF, DynSGS):
       
       DZDX = REFS[15]
       DZDX2 = REFS[16]
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
       DqDt[tdex,2] = DC1[tdex] * P2qPx2[tdex,2]
       DqDt[tdex,3] = DC1[tdex] * P2qPx2[tdex,3]
       
       #%% BOTTOM DIFFUSION (flow along the terrain surface)
       bdex = ebcDex[2]
       dhx = DZDX[bdex,:]
       d2hx = DZDX2[bdex,:]
       dhx2 = np.power(dhx, 2.0)
       S2 = np.reciprocal(dhx2 + 1.0)
       S4 = np.power(S2, 2.0)
       
       # Compute the derivative terms
       P2qPa2 = P2qPx2[bdex,:] + dhx * (P2qPzx[bdex,:] + P2qPxz[bdex,:]) + dhx2 * P2qPz2[bdex,:] + d2hx * PqPz[bdex,:]
       PqPa = PqPx[bdex,:] + dhx * PqPz[bdex,:]
       # Get the coefficients at the ground
       DCB = np.power(S2[:,0] * (np.power(DC1[bdex], 2.0) + np.power(dhx[:,0] * DC2[bdex], 2.0)), 0.5)
       #DCB = DC1[bdex]
       
       dhx = DZDX[bdex,0]
       d2hx = DZDX2[bdex,0]
       dhx2 = np.power(dhx, 2.0)
       S2 = np.reciprocal(dhx2 + 1.0)
       S4 = np.power(S2, 2.0)
       
       # dudt along terrain
       DqDt[bdex,0] = S4 * DCB * ((2.0 + dhx2) * P2qPa2[:,0] + dhx * P2qPa2[:,1] - dhx * d2hx * (PqPa[:,0] + 2.0 * PqPa[:,1]))
       # dwdt along terrain
       DqDt[bdex,1] = dhx * DqDt[bdex,0]
       # dlpdt along terrain
       DqDt[bdex,2] = S2 * DCB * P2qPa2[:,2] - S4 * DCB * dhx * d2hx * (PqPa[:,2])
       # dltdt along terrain
       DqDt[bdex,3] = S2 * DCB * P2qPa2[:,3] - S4 * DCB * dhx * d2hx * (PqPa[:,3])

       # Scale and apply coefficients
       Pr = 0.71 / 0.4
       DqDt[:,3] *= 1.0*Pr
       
       DqDt = enforceTendencyBC(DqDt, zeroDex)
       DqDt = enforceTerrainTendency(DqDt, ebcDex, DZDX)
       
       return DqDt

def computeDiffusiveFluxTendency(PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, REFS, ebcDex, zeroDex, DynSGS):
       
       DZDX = REFS[15]
       DZDX2 = REFS[16]
       DqDt = np.zeros(P2qPx2.shape)
            
       #%% INTERIOR DIFFUSION
       # Diffusion of u-w vector
       DqDt[:,0] = (2.0 * P2qPx2[:,0] + P2qPzx[:,1] + P2qPz2[:,0])  
       DqDt[:,1] = (P2qPx2[:,1] + P2qPxz[:,0] + 2.0 * P2qPz2[:,1])
       # Diffusion of scalars (broken up into anisotropic components
       DqDt[:,2] = (P2qPx2[:,2] + P2qPz2[:,2])
       DqDt[:,3] = (P2qPx2[:,3] + P2qPz2[:,3])
                   
       #%% TOP DIFFUSION (flow along top edge)
       tdex = ebcDex[3]
       DqDt[tdex,0] = 2.0 * P2qPx2[tdex,0]
       DqDt[tdex,2] = P2qPx2[tdex,2]
       DqDt[tdex,3] = P2qPx2[tdex,3]
       
       #%% BOTTOM DIFFUSION (flow along the terrain surface)
       bdex = ebcDex[2]
       dhx = DZDX[bdex,:]
       d2hx = DZDX2[bdex,:]
       dhx2 = np.power(dhx, 2.0)
       S2 = np.reciprocal(dhx2 + 1.0)
       S4 = np.power(S2, 2.0)
       
       # Compute the derivative terms
       P2qPa2 = P2qPx2[bdex,:] + dhx * (P2qPzx[bdex,:] + P2qPxz[bdex,:]) + dhx2 * P2qPz2[bdex,:] + d2hx * PqPz[bdex,:]
       PqPa = PqPx[bdex,:] + dhx * PqPz[bdex,:]
       
       dhx = DZDX[bdex,0]
       d2hx = DZDX2[bdex,0]
       dhx2 = np.power(dhx, 2.0)
       S2 = np.reciprocal(dhx2 + 1.0)
       S4 = np.power(S2, 2.0)
       
       # dudt along terrain
       DqDt[bdex,0] = S4 * ((2.0 + dhx2) * P2qPa2[:,0] + dhx * P2qPa2[:,1] - dhx * d2hx * (PqPa[:,0] + 2.0 * PqPa[:,1]))
       # dwdt along terrain
       DqDt[bdex,1] = dhx * DqDt[bdex,0]
       # dlpdt along terrain
       DqDt[bdex,2] = S2 * P2qPa2[:,2] - S4 * dhx * d2hx * (PqPa[:,2])
       # dltdt along terrain
       DqDt[bdex,3] = S2 * P2qPa2[:,3] - S4 * dhx * d2hx * (PqPa[:,3])

       # Scale and apply coefficients
       Pr = 0.71 / 0.4
       DqDt[:,3] *= 1.0*Pr
       
       DqDt = enforceTendencyBC(DqDt, zeroDex)
       DqDt = enforceTerrainTendency(DqDt, ebcDex, DZDX)
              
       return DqDt