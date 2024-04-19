#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 07:39:25 2019

@author: TempestGuerra
"""

import sys
import math as mt
import numpy as np
from numpy import linalg as lan
import computeDerivativeMatrix as derv

def computePfromSensibleT(DDZ, TZ, AC, P0):
       N = len(TZ)
       
       # Solves for lnP_bar so set the constant of integration
       lnP0 = mt.log(P0)
       
       # Initialize background pressure
       ln_pBar = np.zeros(N)
       
       # Compute the Neumann boundary value at the top z = H
       tempBarI = np.reciprocal(TZ)
       
       #%% Impose BC lnp = 0 @ z = 0 and d(lnp)dz = B @ z = H matched to lnp = -Inf at Inf
       # Specify the derivative at the model top
       dpdZ_H = AC * tempBarI[N-1]
       # Compute adjustment to the derivative matrix operator
       DOP = derv.computeAdjustedOperatorNBC(DDZ, DDZ, -1)

       # Impose resulting Neumann/Dirichlet conditions lnP top and bottom
       NE = N-1
       DOPS = DOP[1:NE,1:NE]
       
       # Index of interior nodes
       idex = range(1,NE)
       # Index of left and interior nodes
       bdex = range(0,NE)
       
       # Compute the forcing due to matching at the model top
       f1 = lnP0 * DOP[:,0]
       f2 = dpdZ_H / DDZ[NE,NE] * DDZ[:,NE]
       F = tempBarI #- f1 - f2
       # Solve the system for lnP
       ln_pBar[idex] = AC * lan.solve(DOPS, (F[1:NE]).astype(dtype=np.float64))
       
       # Compute and set the value at the top that satisfies the BC
       dPdZ_partial = np.dot(DDZ[NE,bdex], ln_pBar[bdex])
       ln_pBar[NE] = (dpdZ_H - dPdZ_partial) / DDZ[NE,NE]
                              
       #%% Reconstruct hydrostatic pressure p from p^K
       ln_pBar += lnP0
       pBar = np.exp(ln_pBar)
       
       return pBar, ln_pBar

def computePfromPotentialT(DDZ, TZ, AC, P0, Kp, N):
       N = len(TZ)
       
       # Initialize background pressure
       pBar = np.zeros(N)
       
       # Compute the Neumann boundary value at the top z = H
       thetaBarI = np.reciprocal(TZ)
       
       #%% Impose BC p^K = 0 @ z = 0 and d(p^K)dz = B @ z = H matched to p^K = 0 at Inf
       # Specify the derivative at the model top
       dpdZ_H = Kp * AC * thetaBarI[N-1]
       # Compute adjustment to the derivative matrix operator
       DOP = derv.computeAdjustedOperatorNBC(DDZ, DDZ, -1)

       # Impose resulting Dirichlet conditions p^K top and bottom
       NE = N-1
       DOPS = DOP[1:NE,1:NE]
       
       # Index of interior nodes
       idex = range(1,NE)
       # Index of left and interior nodes
       bdex = range(0,NE)
       
       # Compute the forcing due to matching at the model top
       f1 = P0 * DDZ[:,0]
       f2 = dpdZ_H / DDZ[NE,NE] * DDZ[:,NE]
       F = thetaBarI - f1 - f2
       # Solve the system for p^K
       pBar[idex] = AC * Kp * lan.solve(DOPS, F[1:NE])
       
       # Compute and set the value at the top that satisfies the BC
       dPdZ_partial = np.dot(DDZ[N-1,bdex], pBar[bdex])
       pBar[N-1] = (dpdZ_H - dPdZ_partial) / DDZ[N-1,N-1]
                              
       #%% Reconstruct hydrostatic pressure p from p^K
       pBar[0:N] += P0**Kp
       pBar = np.power(pBar, 1.0 / Kp)
       
       # Return lnP_bar also
       ln_pBar = np.log(pBar)
       
       return pBar, ln_pBar

def computeThermoMassFields(PHYS, DIMS, REFS, TZ, DTDZ, TempType, isUniform, RLOPT, isStatic):
       
       # Get DIMS data
       NZ = DIMS[4]+1
       
       # Get physical constants (dry air)
       gc = PHYS[0]
       P0 = PHYS[1]
       cp = PHYS[2]
       Rd = PHYS[3]
       Kp = PHYS[4]
       NBVP = PHYS[7]
       
       # Get the reference temperature (uniform profile)
       T0 = TZ[0]
       
       # Get REFS data
       DDZ = REFS[3]
       z = REFS[1]
       #ZTL = REFS[5]
       
       # Solve for background pressure by hydrostatic balance
       if isUniform:
              
              # Analytical potential temperature field
              dlnPTdz = NBVP**2 / gc * np.ones(NZ)
              PT = T0 * np.exp(NBVP**2 / gc * z)
              LPT = np.log(PT)
              
              # Analytical pressure field
              ExP = 1.0 + gc**2 / (cp * NBVP**2 * T0) * (np.exp(-NBVP**2 / gc * z) - 1.0)
              PZ = P0 * np.power(ExP, 1.0 / Kp)
              LPZ = np.log(PZ)
              '''
              # Make profile dry adiabatic near the top boundary (unsupportive of waves)
              if not isStatic:
                     for cc in range(NC):
                            zcol = ZTL[:,cc]
                            zrl = np.argwhere(zcol > (zcol[-1] - 0.5 * RLOPT[0]))
                            
                            dlnPTdz[zrl] = 0.0
                            PT[zrl] = PT[zrl[0]]
                            LPT[zrl] = np.log(PT[zrl])
                            
                            PZ[zrl] = PZ[zrl[0]]
                            LPZ[zrl] = np.log(PZ[zrl])
              '''
              # Log gradient of pressure
              dlnPdz = -(gc / Rd) * np.reciprocal(TZ)
              # Recover density
              RHO = 1.0 / Rd * (PZ * np.reciprocal(TZ))
       else:
              if TempType == 'sensible':
                     AC = - gc / Rd
                     PZ, LPZ = computePfromSensibleT(DDZ, TZ, AC, P0)
                     # Recover vertical gradient in log pressure
                     dlnPdz = AC * np.reciprocal(TZ)
                     # Recover potential temperature background
                     LPT = np.log(TZ) + Rd / cp * (mt.log(P0) - LPZ)
                     dlnPTdz = np.reciprocal(TZ) * DTDZ - Rd / cp * dlnPdz
                     PT = np.exp(LPT)
                     # Recover density
                     RHO = 1.0 / Rd * (PZ * np.reciprocal(TZ))
              elif TempType == 'potential':
                     AC = - gc * P0**Kp / Rd
                     PZ, LPZ = computePfromPotentialT(DDZ, TZ, AC, P0, Kp, NZ)
                     # Recover vertical gradient in log pressure
                     dlnPdz = AC * np.reciprocal(PZ * TZ)
                     # Recover potential temperature background
                     LPT = np.log(TZ)
                     dlnPTdz = np.reciprocal(PT) * DTDZ
                     PT = np.exp(PT)
                     # Recover density
                     RHO = P0**Kp / Rd * np.power(PZ, 1.0 - Kp) * np.reciprocal(TZ)
              else:
                     print('Error: invalid background temperature type chosen.')
                     sys.exit(2)
       
       return dlnPdz, LPZ, PZ, dlnPTdz, LPT, PT, RHO
       
       
              