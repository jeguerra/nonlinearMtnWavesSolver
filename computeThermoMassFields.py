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

def computeAdjustedOperatorNBC(D2A, DOG, DD, tdex, isGivenValue, DP):
       # D2A is the operator to adjust
       # DOG is the original operator to adjust (unadjusted)
       # DD is the 1st derivative operator
       DOP = np.zeros(DD.shape)
       # Get the column span size
       NZ = DD.shape[1]
       cdex = range(NZ)
       cdex = np.delete(cdex, tdex)
       
       # For prescribed value:
       if isGivenValue:
              scale = - DD[tdex,tdex]
       # For matching at infinity
       else:
              scale = (DP - DD[tdex,tdex])
              
       # Loop over columns of the operator and adjust for BC at z = H (tdex)
       for jj in cdex:
              factor = DD[tdex,jj] / scale
              v1 = (D2A[:,jj]).flatten()
              v2 = (DOG[:,tdex]).flatten()
              nvector = v1 + factor * v2
              DOP[:,jj] = nvector
       
       return DOP

def computePfromSensibleT(DDZ, TZ, AC, P0, N):
       # Solves fro lnP_bar so set the constant of integration
       lnP0 = mt.log(P0)
       
       # Initialize background pressure
       ln_pBar = np.zeros(N)
       
       # Compute the Neumann boundary value at the top z = H
       tempBarI = np.reciprocal(TZ)
       
       #%% Impose BC p^K = 0 @ z = 0 and d(p^K)dz = B @ z = H matched to p^K = 0 at Inf
       # Specify the derivative at the model top
       dpdZ_H = AC * tempBarI[N-1]
       # Compute adjustment to the derivative matrix operator
       DOP = computeAdjustedOperatorNBC(DDZ, DDZ, DDZ, N-1, True, None)

       # Impose resulting Dirichlet conditions p^K top and bottom
       NE = N-1
       DOPS = DOP[1:NE,1:NE]
       
       # Index of interior nodes
       idex = range(1,NE)
       # Index of left and interior nodes
       bdex = range(0,NE)
       
       # Compute the forcing due to matching at the model top
       f = -dpdZ_H / DDZ[N-1,N-1] * DDZ[:,N-1]
       F = np.add(tempBarI, f)
       # Solve the system for p^K
       ln_pBar[idex] = AC * lan.solve(DOPS, F[1:NE])
       
       # Compute and set the value at the top that satisfies the BC
       dPdZ_partial = np.dot(DDZ[N-1,bdex], ln_pBar[bdex])
       ln_pBar[N-1] = (dpdZ_H - dPdZ_partial) / DDZ[N-1,N-1]
                              
       #%% Reconstruct hydrostatic pressure p from p^K
       ln_pBar += lnP0
       pBar = np.exp(ln_pBar)
       
       return pBar, ln_pBar

def computePfromPotentialT(DDZ, TZ, AC, P0, Kp, N):
       # Initialize background pressure
       pBar = np.mat(np.zeros(N))
       pBar = pBar.T
       
       # Compute the Neumann boundary value at the top z = H
       thetaBarI = np.reciprocal(TZ)
       
       #%% Impose BC p^K = 0 @ z = 0 and d(p^K)dz = B @ z = H matched to p^K = 0 at Inf
       # Specify the derivative at the model top
       dpdZ_H = Kp * AC * thetaBarI[N-1]
       # Compute adjustment to the derivative matrix operator
       DOP = computeAdjustedOperatorNBC(DDZ, DDZ, DDZ, N-1, True, None)

       # Impose resulting Dirichlet conditions p^K top and bottom
       NE = N-1
       DOPS = DOP[1:NE,1:NE]
       
       # Index of interior nodes
       idex = range(1,NE)
       # Index of left and interior nodes
       bdex = range(0,NE)
       
       # Compute the forcing due to matching at the model top
       f = -dpdZ_H / DDZ[N-1,N-1] * DDZ[:,N-1]
       F = np.add(thetaBarI, f)
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

def computeThermoMassFields(PHYS, DIMS, REFS, TZ, TempType):
       # Get DIMS data
       NZ = DIMS[4]
       
       # Get physical constants (dry air)
       gc = PHYS[0]
       P0 = PHYS[1]
       cp = PHYS[2]
       Rd = PHYS[3]
       Kp = PHYS[4]
       
       # Get REFS data
       DDZ = REFS[7]
       
       # Solve for background pressure by hydrostatic balance
       if TempType == 1:
              AC = - gc / Rd
              PZ, LPZ = computePfromSensibleT(DDZ, TZ, AC, P0, NZ)
              # Recover vertical gradient in log pressure
              dlnPdz = AC * np.reciprocal(TZ)
              # Recover potential temperature background
              LPT = np.log(TZ) + Rd / cp * (mt.log(P0) - LPZ)
              dlnPTdz = np.matmul(DDZ, np.log(TZ)) - Rd / cp * dlnPdz
              PT = np.exp(LPT)
              # Recover density
              RHO = 1.0 / Rd * (PZ * np.reciprocal(TZ))
       elif TempType == 2:
              AC = - gc * P0**Kp / Rd
              PZ, LPZ = computePfromPotentialT(DDZ, TZ, AC, P0, Kp, NZ)
              # Recover vertical gradient in log pressure
              dlnPdz = AC * np.reciprocal(PZ * TZ)
              # Recover potential temperature background
              LPT = np.log(TZ)
              dlnPTdz = np.matmul(DDZ, LPT)
              PT = np.exp(PT)
              # Recover density
              RHO = P0**Kp / Rd * np.power(PZ, 1.0 - Kp) * np.reciprocal(TZ)
       else:
              print('Error: invalid background temperature type chosen.')
              sys.exit(2)
       
       return dlnPdz, LPZ, PZ, dlnPTdz, LPT, PT, RHO
       
       
              