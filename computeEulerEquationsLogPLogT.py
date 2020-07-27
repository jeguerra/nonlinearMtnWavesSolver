#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:11 2019

@author: -
"""
import numpy as np
import scipy.sparse as sps

def computeFieldDerivatives(q, DDX, DDZ, GMLX, GMLZ):
       
       DqDx = DDX.dot(q)
       DqDz = DDZ.dot(q)
       
       DqDx_GML = GMLX.dot(DqDx)
       DqDz_GML = GMLZ.dot(DqDz)
       
       return DqDx, DqDz, DqDx_GML, DqDz_GML

def localDotProduct(arg):
              res = arg[0].dot(arg[1])
              return res

def computePrepareFields(REFS, SOLT, INIT, udex, wdex, pdex, tdex):
       
       TQ = SOLT + INIT
       # Make the total quatities
       U = TQ[udex]
       
       fields = np.reshape(SOLT, (len(udex), 4), order='F')

       return fields, U

#%% Evaluate the Jacobian matrix
def computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, fields, U, botdex, topdex):
       # Get physical constants
       gc = PHYS[0]
       Rd = PHYS[3]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get the derivative operators
       DDXM = REFS[10]
       DDZM = REFS[11]
       DZDX = REFS[15].flatten()
       
       GMLX = REFG[0]
       GMLZ = REFG[1]
       DLTDZ = REFG[2]
       DQDZ = REFG[3]
       
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
       RdT_bar = REFS[9]
       T_bar = (1.0 / Rd) * RdT_bar
       
       bf = np.exp(kap * fields[:,2] + fields[:,3])
       T_ratio = bf - 1.0
       RdT = RdT_bar * bf
       
       # Compute T'
       T_prime = T_ratio * T_bar
       
       RdT_barM = sps.diags(RdT_bar, offsets=0, format='csr')
       RdTM = sps.diags(RdT, offsets=0, format='csr')
       bfM = sps.diags(bf, offsets=0, format='csr')
       
       # Compute derivatives of temperature perturbation
       PtPx = PPXM.dot(T_prime)
       DtDz = DDZM.dot(T_prime)
       
       PtPx = DDXM.dot(T_prime) - DZDX * DtDz
       PtPxM = sps.diags(PtPx, offsets=0, format='csr')
       DtDzM = sps.diags(DtDz, offsets=0, format='csr')
       
       # Compute advective (multiplicative) diagonal operators
       UM = sps.diags(U, offsets=0, format='csr')
       WXZM = sps.diags(WXZ, offsets=0, format='csr')
       
       # Compute common horizontal transport block
       #UPXM = UM.dot(DDXM) + WXZM.dot(DDZM)
       UPXM = UM.dot(GMLX.dot(DDXM)) + WXZM.dot(GMLZ.dot(DDZM))
       
       # Compute the blocks of the Jacobian operator
       LD11 = UPXM + GMLX.dot(PuPxM)
       LD12 = GMLZ.dot(DuDzM + DUDZM)
       LD13 = RdTM.dot(PPXM) + (Rd * PtPxM)
       LD14 = RdTM.dot(PlpPxM)
       
       LD21 = GMLX.dot(PwPxM)
       LD22 = UPXM + GMLZ.dot(DwDzM)
       LD23 = RdTM.dot(DDZM) + RdT_barM.dot(DLTDZM) + Rd * DtDzM
       LD24 = RdTM.dot(DlpDzM) - gc * bfM
       
       LD31 = gam * PPXM + GMLX.dot(PlpPxM)
       LD32 = gam * DDZM + GMLZ.dot(DlpDzM + DLPDZM)
       LD33 = UPXM
       LD34 = None
       
       LD41 = GMLX.dot(PltPxM)
       LD42 = GMLZ.dot(DltDzM + DLPTDZM)
       LD43 = None
       LD44 = UPXM
       
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
       PORZ = REFS[9]
       # Full spectral transform derivative matrices
       DDXM = REFS[10]
       DDZM = REFS[11]
              
       #%% Compute the various blocks needed
       UM = sps.diags(UZ, offsets=0, format='csr')
       PORZM = sps.diags(PORZ, offsets=0, format='csr')
       
       # Compute hydrostatic state diagonal operators
       DLTDZ = REFG[2]
       DQDZ = REFG[3]
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

# Function evaluation of the non linear equations (dynamic components)
def computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, DZDX, RdT_bar, fields, U):
       # Get physical constants
       gc = PHYS[0]
       kap = PHYS[4]
       gam = PHYS[6]
       
       # Get hydrostatic initial fields
       DQDZ = REFG[3]
       DQDZ_GML = REFG[1].dot(REFG[3])

       # Compute the partial derivative
       PqPx = DqDx - DZDX * DqDz
       PqPx_GML = DqDx_GML - DZDX * DqDz_GML
       
       # Compute advective (multiplicative) operators
       UM = np.expand_dims(U,1)
       wxz = np.expand_dims(fields[:,1],1)
       
       # Compute pressure gradient force scaling (buoyancy)
       T_ratio = np.exp(kap * fields[:,2] + fields[:,3]) - 1.0
       RdT = RdT_bar * (1.0 + T_ratio)
       
       # Compute transport and divergence terms
       #UPqPx = UM * PqPx
       #wDQqDz = wxz * (DqDz + DQDZ)
       
       UPqPx = UM * PqPx_GML
       wDQqDz = wxz * (DqDz_GML + DQDZ_GML)
       transport = UPqPx + wDQqDz
       
       divergence = (PqPx[:,0] + DqDz[:,1])
       #divergence = (PqPx_GML[:,0] + DqDz_GML[:,1])
       
       DqDt = -transport
       # Horizontal momentum equation
       DqDt[:,0] -= RdT * PqPx[:,2]
       # Vertical momentum equation
       DqDt[:,1] -= (RdT * DqDz[:,2] - gc * T_ratio)
       # Pressure (mass) equation
       DqDt[:,2] -= gam * divergence
       # Potential Temperature equation (transport only)
                                  
       return DqDt

def computeRayleighTendency(REFG, fields):
       
       # Get the Rayleight operators
       mu = np.expand_dims(REFG[4],0)
       ROP = REFG[5]
       
       DqDt = -mu * ROP.dot(fields)
       
       return DqDt

def computeDiffusiveFluxTendency(RESCF, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex):
       
       # Get the anisotropic coefficients
       RESCFX = RESCF[0]
       RESCFZ = RESCF[1]
       
       # Compute the partial derivative
       PqPx = DqDx - DZDX * DqDz
       
       # Compute diffusive fluxes
       xflux = RESCFX * PqPx
       xflux[:,0] *= 2.0
       zflux = RESCFZ * DqDz
       zflux[:,1] *= 2.0
              
       # Compute derivatives of fluxes
       DDxx = DDXM.dot(xflux)
       DDxz = DDZM.dot(xflux)
       PPx2 = DDxx - (DZDX * DDxz)
       DDz2 = DDZM.dot(zflux)
       
       # Apply Neumann condition
       PPx2[ebcDex[1],:] *= 0.0
       DDz2[ebcDex[2],:] *= 0.0
       
       # Compute the tendencies (divergence of diffusive flux... discontinuous)
       DqDt = PPx2 + DDz2
       
       return DqDt

def computeDiffusionTendency(RESCF, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex):
       
       # Get the anisotropic coefficients
       RESCFX = RESCF[0]
       RESCFZ = RESCF[1]
       
       # Compute the partial derivative
       PqPx = DqDx - DZDX * DqDz
              
       # Compute 2nd partials of perturbations
       DDxx = DDXM.dot(PqPx)
       DDxz = DDZM.dot(PqPx)
       
       PPx2 = DDxx - (DZDX * DDxz)
       DDz2 = DDZM.dot(DqDz)
       
       # Compute diffusive fluxes
       xflux = RESCFX * PPx2
       xflux[:,0] *= 2.0
       zflux = RESCFZ * DDz2
       zflux[:,1] *= 2.0
       
       # Apply Neumann condition
       xflux[ebcDex[1],:] *= 0.0
       zflux[ebcDex[2],:] *= 0.0
       
       # Compute the tendencies
       DqDt = xflux + zflux
       
       return DqDt
       