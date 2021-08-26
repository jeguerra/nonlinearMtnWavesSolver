#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:59:22 2019

@author: TempestGuerra
"""

import scipy.linalg as scl
import numpy as np
import math as mt
from computeGrid import computeGrid
from HerfunChebNodesWeights import hefuncm, hefunclb
from HerfunChebNodesWeights import chebpolym, cheblb
from HerfunChebNodesWeights import lgfuncm, lgfunclb
from HerfunChebNodesWeights import legpolym, leglb

def computeAdjustedOperatorNBC(D2A, DOG, DD, tdex):
       # D2A is the operator to adjust
       # DOG is the original operator to adjust (unadjusted)
       # DD is the 1st derivative operator
       DOP = np.zeros(DD.shape)
       # Get the column span size
       NZ = DD.shape[1]
       cdex = range(NZ)
       cdex = np.delete(cdex, tdex)
       
       scale = - DD[tdex,tdex]       
       # Loop over columns of the operator and adjust for BC some location tdex
       for jj in cdex:
              factor = DD[tdex,jj] / scale
              v1 = (D2A[:,jj]).flatten()
              v2 = (DOG[:,tdex]).flatten()
              nvector = v1 + factor * v2
              DOP[:,jj] = nvector
       
       return DOP

def computeAdjustedOperatorPeriodic(D2A):
       # D2A is the operator to adjust
       print('@@ PERIODIC BC @@')
       DOP = np.copy(D2A)
       
       # Connect leading and trailing equations
       DOP[0,:] += D2A[-1,:]
       DOP[-1,:] += D2A[0,:]
       DOP[0,:] *= 0.5
       DOP[-1,:] *= 0.5
       
       return DOP

def computeAdjustedOperatorNBC2(D2A, DOG, DD, tdex, isGivenValue, DP):
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
              scale = -DD[tdex,tdex]
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
       
       # Here DP works as the coefficient to the prescribed derivative
       if not isGivenValue and abs(DP) > 0.0:
              DOP[tdex,tdex] += DP
       
       return DOP

def numericalCleanUp(DDM):
       
       N = DDM.shape
       ZTOL = 1.0E-16
       DDMC = np.copy(DDM)
       # Clean up numerical zeros
       for ii in range(N[0]):
              for jj in range(N[1]):
                     if abs(DDM[ii,jj]) <= ZTOL:
                            DDMC[ii,jj] = 0.0
       return DDMC

# Compute Spectral Element 1st derivative matrix
def computeSpectralElementDerivativeMatrix(dom, NE, isLaguerreEnds, resolutions):
       N = len(dom)
       
       # Get the resolution multipliers
       if resolutions != None:
              EF0 = resolutions[0]
              EF1 = resolutions[1]
       else:
              EF0 = 1.0; EF1 = 1.0
              
       if isLaguerreEnds:
              print('@@ USING Laguerre SEM at BC @@')

       # Master grid
       isLegendre = True
       isChebyshev = False
       if isLegendre:
              print("@@ LEGENDRE SEM @@")
              grid = [True, False, False, True, False]
       
       if isChebyshev:
              print('@@ CHEBYSHEV SEM @@')
              grid = [True, False, True, False, False]
       
       endGrid = [True, False, False, False, True]
       
       emats = []
       MS = []
       LE = []
       
       # Loop over each element in dom (N-1 in 1D)
       for ee in range(N-1):
              # Get the element coordinates
              xa = dom[ee]
              xb = dom[ee+1]
              LE += [abs(xb - xa)]
              
              # Compute the local cubic-spline derivative matrix
              if ee == 0:
                     MS += [int(EF1 * NE)]
                     SDIMS = [xa, xb, LE[ee], MS[ee], MS[ee]]
                     if isLaguerreEnds:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom = -1.0 * np.flip(SREFS[1] + abs(xb))
                            DMB, CMB, scale, wlg1 = computeLaguerreDerivativeMatrix(SDIMS)
                            DMS = -1.0 * np.flip(np.flip(DMB,axis=0), axis=1)
                     else:
                            SREFS = computeGrid(SDIMS, *grid)
                            sdom = SREFS[1] + xa
                            if isLegendre:
                                   DMS, CMB = computeLegendreDerivativeMatrix(SDIMS)
                            if isChebyshev:
                                   DMS, CMB = computeChebyshevDerivativeMatrix(SDIMS)
                     
              elif ee == N-2:
                     MS += [int(EF1 * NE)]
                     SDIMS = [xa, xb, LE[ee], MS[ee], MS[ee]]                     
                     if isLaguerreEnds:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom = SREFS[1] + xa
                            DMB, CMB, scale, wlg2 = computeLaguerreDerivativeMatrix(SDIMS)
                            DMS = 1.0 * DMB
                     else:
                            SREFS = computeGrid(SDIMS, *grid)
                            sdom = SREFS[1] + xa
                            if isLegendre:
                                   DMS, CMB = computeLegendreDerivativeMatrix(SDIMS)
                            if isChebyshev:
                                   DMS, CMB = computeChebyshevDerivativeMatrix(SDIMS)
              else:
                     MS += [int(EF0 * NE)]
                     SDIMS = [xa, xb, LE[ee], MS[ee], MS[ee]]
                     SREFS = computeGrid(SDIMS, *grid)
                     sdom = SREFS[1] + xa
                     if isLegendre:
                            DMS, CMB = computeLegendreDerivativeMatrix(SDIMS)
                     if isChebyshev:
                            DMS, CMB = computeChebyshevDerivativeMatrix(SDIMS)
              
              # Make a list of element matrices
              emats += [DMS]
              
              if ee == 0:
                     edom = sdom
              else:
                     edom = np.append(edom, sdom[1:])
                     
       # Loop over each dof and assemble the global matrix
       for ee in range(N-1):
                     
              bdex = -(MS[ee]+2)
              tdex = -(MS[ee]+1)
                            
              if ee == 0:
                     DDMS = emats[ee]
              else:
                     # Append the next matrix
                     DDMS = scl.block_diag(DDMS, emats[ee])
                     
                     # Combine coincident row/col
                     acol = np.delete(DDMS[:,bdex], tdex)
                     bcol = np.delete(DDMS[:,tdex], bdex)
                     arow = np.delete(DDMS[bdex,:], tdex)
                     brow = np.delete(DDMS[tdex,:], bdex)
                     # Scaling of matched coefficients here
                     LM = (LE[ee-1] + LE[ee])
                     asc = LE[ee-1] / LM
                     bsc = LE[ee] / LM
                     
                     mrow = (asc * arow + bsc * brow)
                     #mrow = 0.5 * (arow + brow)
                     mcol = (1.0 * acol + 1.0 * bcol)
                     
                     # Delete redundant row/col
                     DDMS = np.delete(DDMS, tdex, axis=0)
                     DDMS = np.delete(DDMS, tdex, axis=1)
                     
                     # Set the merged row/col
                     DDMS[:,tdex] = mcol
                     DDMS[tdex,:] = mrow
       
       DDMSA = numericalCleanUp(DDMS)
       
       return DDMSA, edom

# Compute Spectral Cubic Spline Element 1st derivative matrix
def computeSCSElementDerivativeMatrix(dom, NE, isLaguerreEnds, resolutions):
       N = len(dom)
       
       # Get the resolution multipliers
       if resolutions != None:
              EF0 = resolutions[0]
              EF1 = resolutions[1]
       else:
              EF0 = 1.0; EF1 = 1.0
       
       isClampedEnds = True
       
       if isClampedEnds:
              rightEnd = [True, False, False, False]
              leftEnd = [True, False, False, False]
       else:
              rightEnd = [False, False, False, True]
              leftEnd = [False, False, True, False]
       
       # Interior configuration for SCSE
       interior = [True, False, False, False]
       # Master grid
       grid = [True, False, False, True, False]
       endGrid = [True, False, False, False, True]
       
       emats = []
       MS = []
       LE = []
       
       # Loop over each element in dom (N-1 in 1D)
       for ee in range(N-1):
              # Get the element coordinates
              xa = dom[ee]
              xb = dom[ee+1]
              LE += [abs(xb - xa)]
              
              # Compute the local cubic-spline derivative matrix
              if ee == 0:
                     MS += [int(EF1 * NE)]
                     SDIMS = [xa, xb, LE[ee], MS[ee], MS[ee]]
                     if isLaguerreEnds:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom = -1.0 * np.flip(SREFS[1] + abs(xb))
                            DMB, CMB, scale, wlg1 = computeLaguerreDerivativeMatrix(SDIMS)
                            DMB_back = -1.0 * np.flip(np.flip(DMB,axis=0), axis=1)
                            DMS, DMS2 = computeCubicSplineDerivativeMatrix(sdom, *interior, DMB_back)
                     else:
                            SREFS = computeGrid(SDIMS, *grid)
                            sdom = SREFS[1] + xa
                            DMB, CMB = computeLegendreDerivativeMatrix(SDIMS)
                            DMS, DMS2 = computeCubicSplineDerivativeMatrix(sdom, *interior, DMB)
                     
              elif ee == N-2:
                     MS += [int(EF1 * NE)]
                     SDIMS = [xa, xb, LE[ee], MS[ee], MS[ee]]                     
                     if isLaguerreEnds:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom = SREFS[1] + xa
                            DMB, CMB, scale, wlg2 = computeLaguerreDerivativeMatrix(SDIMS)
                            DMB_fwrd = 1.0 * DMB
                            DMS, DMS2 = computeCubicSplineDerivativeMatrix(sdom, *interior, DMB_fwrd)
                     else:
                            SREFS = computeGrid(SDIMS, *grid)
                            sdom = SREFS[1] + xa
                            DMB, CMB = computeLegendreDerivativeMatrix(SDIMS)
                            DMS, DMS2 = computeCubicSplineDerivativeMatrix(sdom, *interior, DMB)
              else:
                     MS += [int(EF0 * NE)]
                     SDIMS = [xa, xb, LE[ee], MS[ee], MS[ee]]
                     SREFS = computeGrid(SDIMS, *grid)
                     sdom = SREFS[1] + xa
                     DMB, CMS = computeLegendreDerivativeMatrix(SDIMS)
                     DMS, DMS2 = computeCubicSplineDerivativeMatrix(sdom, *interior, DMB)
              
              # Make a list of element matrices
              #print(DMS)
              emats += [DMS]
              
              if ee == 0:
                     edom = sdom
              else:
                     edom = np.append(edom, sdom[1:])
                     
       # Loop over each dof and assemble the global matrix
       for ee in range(N-1):
                     
              bdex = -(MS[ee]+2)
              tdex = -(MS[ee]+1)
                            
              if ee == 0:
                     DDMS = emats[ee]
              else:
                     # Append the next matrix
                     DDMS = scl.block_diag(DDMS, emats[ee])
                     
                     # Combine coincident row/col
                     acol = np.delete(DDMS[:,bdex], tdex)
                     bcol = np.delete(DDMS[:,tdex], bdex)
                     mcol = (acol + bcol)
                     arow = np.delete(DDMS[bdex,:], tdex)
                     brow = np.delete(DDMS[tdex,:], bdex)
                     # Scaling of matched coefficients here
                     LM = (LE[ee-1] + LE[ee])
                     asc = LE[ee-1] / LM
                     bsc = LE[ee] / LM
                     mrow = (asc * arow + bsc * brow)
                     
                     # Delete redundant row/col
                     DDMS = np.delete(DDMS, tdex, axis=0)
                     DDMS = np.delete(DDMS, tdex, axis=1)
                     
                     # Set the merged row/col
                     DDMS[:,tdex] = mcol
                     DDMS[tdex,:] = mrow
                     
       DDMSA = numericalCleanUp(DDMS)
       
       return DDMSA, edom

# Computes Cubic Spline 1st derivative matrix
def computeCubicSplineDerivativeMatrix(dom, isClamped, isEssential, \
                                       isLeftEssentialRightClamped, isLeftClampedRightEssential, DDM_BC):
       # Initialize matrix blocks
       N = len(dom)
       A = np.zeros((N,N)) # coefficients to 2nd derivatives
       B = np.zeros((N,N)) # coefficients to RHS of 2nd derivatives
       C = np.zeros((N,N)) # coefficients to 1st derivatives
       D = np.zeros((N,N)) # coefficients to additive part of 1st derivatives
       
       # Loop over each interior point in the irregular grid
       for ii in range(1,N-1):
              hp = abs(dom[ii+1] - dom[ii])
              hm = abs(dom[ii] - dom[ii-1])
              hc = abs(dom[ii+1] - dom[ii-1])
              
              A[ii,ii-1] = -1.0 / 6.0 * hm
              A[ii,ii] = -1.0 / 3.0 * hc
              A[ii,ii+1] = -1.0 / 6.0 * hp
              
              B[ii,ii-1] = -1.0 / hm
              B[ii,ii] = (1.0 / hm + 1.0 / hp)
              B[ii,ii+1] = -1.0 / hp
              
       for ii in range(1,N-1):
              hp = abs(dom[ii+1] - dom[ii])
              C[ii,ii] = -1.0 / 3.0 * hp
              C[ii,ii+1] = -1.0 / 6.0 * hp
              
              D[ii,ii] = -1.0 / hp
              D[ii,ii+1] = 1.0 / hp
              
       # Compute BC adjustments
       h0 = abs(dom[1] - dom[0])
       hn = abs(dom[N-1] - dom[N-2])
       
       if isClamped:
              # Left end
              A[0,0] = -2.0 * h0 / 3.0
              A[0,1] = h0 / 6.0
              
              # Use derivative by CFD to set boundary condition
              B[0,:] = DDM_BC[0,:]
              B[0,0] += 1.0 / h0
              B[0,1] -= 1.0 / h0
              
              # Right end
              A[N-1,N-2] = -hn / 6.0
              A[N-1,N-1] = 2.0 * hn / 3.0
              
              # Use derivative by CFD to set boundary condition
              B[N-1,:] = DDM_BC[N-1,:]
              B[N-1,N-2] += 1.0 / hn
              B[N-1,N-1] -= 1.0 / hn
              #'''
              
              # Compute the first derivative matrix
              AIB = np.linalg.solve(A, B)
              DDM = C.dot(AIB) + D
              
              # Adjust ends
              DDM[0,:] = DDM_BC[0,:]
              DDM[N-1,:] = DDM_BC[N-1,:]
              
       elif isEssential:
              # Left end
              A[0,0] = -1.0 / h0
              A[0,1] = 1.0 / h0
              B[0,:] = np.zeros(N)
              
              # Right end
              A[N-1,N-2] = 1.0 / hn
              A[N-1,N-1] = -1.0 / hn
              B[N-1,:] = np.zeros(N)
              
              # Compute the first derivative matrix
              AIB = np.linalg.solve(A, B)
              DDM = C.dot(AIB) + D
              
              # Adjust the ends
              DDM[0,:] = h0 / 6.0 * AIB[1,:] - 2.0 * h0 / 3.0 * AIB[0,:]
              DDM[0,0] -= 1.0 / h0
              DDM[0,1] += 1.0 / h0
              
              DDM[N-1,:] = -h0 / 6.0 * AIB[N-2,:] + 2.0 * hn / 3.0 * AIB[N-1,:]
              DDM[N-1,N-2] -= 1.0 / hn
              DDM[N-1,N-1] += 1.0 / hn
       
       elif isLeftEssentialRightClamped:
              # Left end
              A[0,0] = -1.0 / h0
              A[0,1] = 1.0 / h0
              B[0,:] = np.zeros(N)
              
              # Right end
              A[N-1,N-2] = -hn / 6.0
              A[N-1,N-1] = 2.0 * hn / 3.0
              
              # Use derivative by CFD to set boundary condition
              B[N-1,:] = DDM_BC[N-1,:]
              B[N-1,N-2] += 1.0 / hn
              B[N-1,N-1] -= 1.0 / hn
              
              # Compute the first derivative matrix
              AIB = np.linalg.solve(A, B)
              DDM = C.dot(AIB) + D
              #'''
              # Adjust ends
              DDM[0,:] = h0 / 6.0 * AIB[1,:] - 2.0 * h0 / 3.0 * AIB[0,:]
              DDM[0,0] -= 1.0 / h0
              DDM[0,1] += 1.0 / h0
              
              DDM[N-1,:] = 1.0 * DDM_BC[N-1,:]
              #'''
       elif isLeftClampedRightEssential:
              # Left end
              A[0,0] = -2.0 * h0 / 3.0
              A[0,1] = h0 / 6.0
              
              # Use derivative by CFD to set boundary condition
              B[0,:] = DDM_BC[0,:]
              B[0,0] += 1.0 / h0
              B[0,1] -= 1.0 / h0
              
              # Right end
              A[N-1,N-2] = 1.0 / hn
              A[N-1,N-1] = -1.0 / hn
              B[N-1,:] = np.zeros(N)
              
              # Compute the first derivative matrix
              AIB = np.linalg.solve(A, B)
              DDM = C.dot(AIB) + D
              #'''
              # Adjust ends
              DDM[0,:] = 1.0 * DDM_BC[0,:]
              
              DDM[N-1,:] = -h0 / 6.0 * AIB[N-2,:] + 2.0 * hn / 3.0 * AIB[N-1,:]
              DDM[N-1,N-2] -= 1.0 / hn
              DDM[N-1,N-1] += 1.0 / hn
              #'''
       else:
              # NATURAL cubic spline.
              AIB = np.zeros((N,N))
              AIB[1:N-1,1:N-1] = np.linalg.solve(A[1:N-1,1:N-1], B[1:N-1,1:N-1])
              DDM = C.dot(AIB) + D
              
              # Adjust the ends
              DDM[0,:] = h0 / 6.0 * AIB[1,:]
              DDM[0,0] -= 1.0 / h0
              DDM[0,1] += 1.0 / h0
              
              DDM[N-1,:] = -h0 / 6.0 * AIB[N-2,:]
              DDM[N-1,N-2] -= 1.0 / hn
              DDM[N-1,N-1] += 1.0 / hn      
              
       DDMC = numericalCleanUp(DDM)

       return DDMC, AIB

# Computes standard 4th order compact finite difference 1st derivative matrix
def computeCompactFiniteDiffDerivativeMatrix1(DIMS, dom):
       # Initialize the left and right derivative matrices
       N = len(dom)
       LDM = np.zeros((N,N)) # tridiagonal
       RDM = np.zeros((N,N)) # centered difference
       
       # Loop over each interior point in the irregular grid
       for ii in range(1,N-1):
              # Set compact finite difference
              # Get the metric weights
              hp = dom[ii+1] - dom[ii]
              hm = dom[ii] - dom[ii-1]
              
              # Compute the stencil coefficients
              hr = (hm / hp)
              d = -0.25
              c = d * hr**4
              b = -1.0 / 8.0 * (5.0 + hr) 
              a = 1.0 / 8.0 * (hr**2 + hr**3) + 0.5 * hr**4
              
              # Write the right equation
              RDM[ii,ii-1] = -b
              RDM[ii,ii] = (a + b)
              RDM[ii,ii+1] = -a
              # Write the left equation
              LDM[ii,ii-1] = d * hm
              LDM[ii,ii] = -(hp * (a + c) + hm * (d - b))
              LDM[ii,ii+1] = c * hp
                     # Handle the left and right boundaries
       LDM[0,0] = 1.0
       LDM[N-1,N-1] = 1.0
       
       # Left end (forward)
       hp = dom[1] - dom[0]
       hpp = hp + (dom[2] - dom[1])
       lc = (hp - (hp**2 / hpp))
       RDM[0,0] = -(1.0 / lc) * (1.0 - (hp / hpp)**2)
       RDM[0,1] = (1.0 / lc)
       RDM[0,2] = -(1.0 / lc) * (hp / hpp)**2
       
       # Right end (backward)
       hm = dom[N-2] - dom[N-1]
       hmm = hm + (dom[N-3] - dom[N-2])
       rc = (hm - (hm**2 / hmm))
       RDM[N-1,N-1] = -(1.0 / rc) * (1.0 - (hm / hmm)**2)
       RDM[N-1,N-2] = (1.0 / rc)
       RDM[N-1,N-3] = -(1.0 / rc) * (hm / hmm)**2

       # Get the derivative matrix
       DDM1 = np.linalg.solve(LDM, RDM)
       
       DDM1C = numericalCleanUp(DDM1)
       
       return DDM1C

# Computes standard 4th order compact finite difference 2nd derivative matrix
def computeCompactFiniteDiffDerivativeMatrix2(DIMS, dom):
       # Initialize the left and right derivative matrices
       N = len(dom)
       LDM = np.zeros((N,N)) # tridiagonal
       RDM = np.zeros((N,N)) # centered difference
       
       # Loop over each interior point in the irregular grid
       for ii in range(1,N-1):
              # Set compact finite difference
              # Get the metric weights
              hp = dom[ii+1] - dom[ii]
              hm = dom[ii] - dom[ii-1]
              
              # Compute the stencil coefficients
              hr = (hm / hp)
              d = 3.0 / 24.0 - 1.0 / 24.0 * (1.0 / hr)**3
              c = 3.0 / 24.0 - 1.0 / 24.0 * hr**3
              b = 1.0
              a = hr
              
              # Write the right equation
              RDM[ii,ii-1] = b
              RDM[ii,ii] = -(a + b)
              RDM[ii,ii+1] = a
              # Write the left equation
              LDM[ii,ii-1] = -d * hm**2
              LDM[ii,ii] = ((0.5 * a + c) * hp**2 + (0.5 * b + d) * hm**2) 
              LDM[ii,ii+1] = -c * hp**2
              
       # Handle the left and right boundaries
       LDM[0,0] = 1.0
       LDM[N-1,N-1] = 1.0
       
       # Left end (forward)
       hp = dom[1] - dom[0]
       hpp = hp + (dom[2] - dom[1])
       hr = hp / hpp
       cd = 0.5 * (hp * (dom[2] - dom[1]))
       RDM[0,0] = (1.0 - hr) / cd 
       RDM[0,1] = -1.0 / cd
       RDM[0,2] = hr / cd
       
       # Right end (backward)
       hm = dom[N-2] - dom[N-1]
       hmm = hm + (dom[N-3] - dom[N-2])
       hr = hm / hmm
       cd = 0.5 * (hm * (dom[N-3] - dom[N-2]))
       RDM[N-1,N-1] = (1.0 - hr) / cd
       RDM[N-1,N-2] = -1.0 / cd
       RDM[N-1,N-3] = hr / cd

       # Get the derivative matrix
       DDM2 = np.linalg.solve(LDM, RDM)
       
       DDM2C = numericalCleanUp(DDM2)

       return DDM2C

def computeHermiteFunctionDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       L1 = DIMS[0]
       L2 = DIMS[1]
       NX = DIMS[3]
       
       alpha, whf = hefunclb(NX)
       HT = hefuncm(NX, alpha, True)
       HTD = hefuncm(NX+1, alpha, True)
       
       # Get the scale factor
       b = (np.amax(alpha) - np.min(alpha)) / abs(L2 - L1)
       
       # Make a diagonal matrix of weights
       W = np.diag(whf, k=0)
       
       # Compute the coefficients of spectral derivative in matrix form
       SDIFF = np.zeros((NX+2,NX+1))
       SDIFF[0,1] = mt.sqrt(0.5)
       SDIFF[NX,NX-1] = -mt.sqrt(NX * 0.5)
       SDIFF[NX+1,NX] = -mt.sqrt((NX + 1) * 0.5)
                     
       for rr in range(1,NX):
              SDIFF[rr,rr+1] = mt.sqrt((rr + 1) * 0.5)
              SDIFF[rr,rr-1] = -mt.sqrt(rr * 0.5)
              
       # Hermite function spectral transform in matrix form
       STR_H = HT.dot(W)
       # Hermite function spatial derivative based on spectral differentiation
       temp = (HTD.T).dot(SDIFF)
       temp = temp.dot(STR_H)
       DDM = b * temp
       
       DDMC = numericalCleanUp(DDM)
       
       return DDMC.astype(np.float64), STR_H

def computeChebyshevDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       # Initialize grid and make column vector
       xi, wcp = cheblb(NZ)
   
       # Get the Chebyshev transformation matrix
       CT = chebpolym(NZ+1, -xi)
   
       # Make a diagonal matrix of weights
       W = np.diag(wcp)
   
       # Compute scaling for the forward transform
       S = np.eye(NZ+1)
   
       for ii in range(NZ):
              temp = W.dot(CT[:,ii])
              temp = ((CT[:,ii]).T).dot(temp)
              S[ii,ii] = temp ** (-1)

       S[NZ,NZ] = 1.0 / mt.pi
   
       # Compute the spectral derivative coefficients
       SDIFF = np.zeros((NZ+1,NZ+1))
       SDIFF[NZ-1,NZ] = 2.0 * NZ
   
       for ii in reversed(range(NZ - 1)):
              A = 2.0 * (ii + 1)
              B = 1.0
              if ii > 0:
                     c = 1.0
              else:
                     c = 2.0
            
              SDIFF[ii,:] = B / c * SDIFF[ii+2,:]
              SDIFF[ii,ii+1] = A / c
    
       # Chebyshev spectral transform in matrix form
       temp = CT.dot(W)
       STR_C = S.dot(temp)
       # Chebyshev spatial derivative based on spectral differentiation
       # Domain scale factor included here
       temp = (CT.T).dot(SDIFF)
       DDM = -(2.0 / ZH) * temp.dot(STR_C)
       
       DDMC = numericalCleanUp(DDM)
       
       #print(xi)
       #print(DDM[0,0], -(2.0 * NZ**2 + 1) / 3.0 / ZH)
       #print(DDM[-1,-1], (2.0 * NZ**2 + 1) / 3.0 / ZH)

       return DDMC, STR_C

def computeFourierDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       L1 = DIMS[0]
       L2 = DIMS[1]
       NX = DIMS[3]
       
       kxf = (2*mt.pi/abs(L2 - L1)) * np.fft.fftfreq(NX+1) * (NX+1)
       KDM = np.diag(kxf, k=0)
       DFT = np.fft.fft(np.eye(NX+1), axis=0)
       DDM = np.fft.ifft(1j * KDM.dot(DFT), axis=0)
       
       DDMC = numericalCleanUp(DDM)
       
       return DDMC, DFT

def computeLaguerreDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       xi, wlf = lgfunclb(NZ)
       LT = lgfuncm(NZ, xi, True)
              
       # Get the scale factor
       b = np.amax(xi) / abs(ZH)
       
       # Make a diagonal matrix of weights
       W = np.diag(wlf, k=0)
       
       # Compute the spectral derivative coefficients
       SDIFF = np.zeros((NZ+1,NZ+1))
       SDIFF[NZ,NZ] = -0.5
                   
       for rr in reversed(range(NZ)):
              SDIFF[rr,:] = SDIFF[rr+1,:]
              SDIFF[rr,rr+1] -= 0.5
              SDIFF[rr,rr] -= 0.5
              
       # Hermite function spectral transform in matrix form
       STR_L = (LT).dot(W)
       # Hermite function spatial derivative based on spectral differentiation
       temp = (LT.T).dot(SDIFF)
       temp = temp.dot(STR_L)
       #print(temp[0,0], -NZ / 2 - 0.5)
       
       lead = -NZ / 2 - 0.5
       if temp[0,0] != lead:
              temp[0,0] = lead
       
       DDM = b * temp
       
       DDMC = numericalCleanUp(DDM)
       
       return DDMC.astype(np.float64), STR_L, b, wlf

def computeLegendreDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       xi, wlf = leglb(NZ)
       LT, DLT = legpolym(NZ, xi, True)
       
       # Make a diagonal matrix of weights
       W = np.diag(wlf, k=0)
       
       # Compute scaling for the forward transform
       S = np.eye(NZ+1)
   
       for ii in range(NZ):
              S[ii,ii] = ii + 0.5
       S[NZ,NZ] = 0.5 * NZ
       
       # Compute the spectral derivative coefficients
       SDIFF = np.zeros((NZ+1,NZ+1))
       SDIFF[NZ,NZ] = 0.0
       SDIFF[NZ-1,NZ] = (2 * NZ - 1)
                   
       for rr in reversed(range(1,NZ)):
              A = (2 * rr - 1)
              B = (2 * rr + 3)
              SDIFF[rr-1,rr] = A
              SDIFF[rr-1,:] += (A / B) * SDIFF[rr+1,:]
              
       # Legendre spectral transform in matrix form
       temp = (LT).dot(W)
       STR_L = S.dot(temp)
       # Legendre spatial derivative based on spectral differentiation
       # Domain scale factor included here
       temp = (LT.T).dot(SDIFF)
       DDM = (2.0 / ZH) * temp.dot(STR_L)
       
       DDMC = numericalCleanUp(DDM)
       
       #print(DDM[0,0], -NZ * (NZ + 1) / 2.0 / ZH)
       #print(DDM[-1,-1], NZ * (NZ + 1) / 2.0 / ZH)
       
       return DDMC, STR_L
       
       