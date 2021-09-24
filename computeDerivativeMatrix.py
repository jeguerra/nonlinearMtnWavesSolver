#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:59:22 2019

@author: TempestGuerra
"""

import scipy.linalg as scl
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from computeGrid import computeGrid
from HerfunChebNodesWeights import hefuncm, hefunclb
from HerfunChebNodesWeights import chebpolym, cheblb
from HerfunChebNodesWeights import lgfuncm, lgfunclb
from HerfunChebNodesWeights import legpolym, leglb

def computeAdjustedOperatorNBC(D2A, DD, tdex):
       # D2A is the operator to adjust
       # DD is the 1st derivative operator
       R = -DD[tdex,tdex]
       DA = (1.0 / R) * np.outer(DD[:,tdex], DD[tdex,:])
       
       DOP = D2A + DA
       DOP[tdex,:] = 0.0
       DOP[:,tdex] = 0.0
       
       return DOP

def computeAdjustedOperatorNBC_ends(D2A, DD):
       # D2A is the operator to adjust
       # DD is the 1st derivative operator
       
       R = (DD[0,0] * DD[-1,-1] - DD[0,-1] * DD[-1,0])
       
       V1 = (DD[-1,0] / R) * DD[:,-1] - (DD[-1,-1] / R) * DD[:,0] 
       DA1 = np.outer(V1, DD[0,:])
       
       V2 = (DD[0,-1] / R) * DD[:,0] - (DD[0,0] / R) * DD[:,-1] 
       DA2 = np.outer(V2, DD[-1,:])
       
       DOP = D2A + DA1 + DA2
       
       DOPC = numericalCleanUp(DOP)
       
       return DOPC

def computeAdjustedOperatorPeriodic(D2A):
       # D2A is the operator to adjust
       print('@@ PERIODIC BC @@')
       DOP = np.zeros(D2A.shape)
       NC = D2A.shape[1]
       
       # Copy over interior columns
       for jj in range(1,NC-1):
              DOP[:,jj] = D2A[:,jj] 
              
       # Couple the columns
       DOP[:,0] += D2A[:,-1]
       DOP[:,-1] += D2A[:,0]
       
       # Couple the rows and scale
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
       ZTOL = 1.0E-15
       DDMC = np.copy(DDM)
       # Clean up numerical zeros
       for ii in range(N[0]):
              for jj in range(N[1]):
                     if abs(DDM[ii,jj]) <= ZTOL:
                            DDMC[ii,jj] = 0.0
       return DDMC

# Compute Spectral Element 1st derivative matrix
def computeSpectralElementDerivativeMatrix(dom, NE, adjustedMethod):
       
       #commonDerivativeMethod = 'interface_average'
       #commonDerivativeMethod = 'order4_compactFD'
       commonDerivativeMethod = 'order6_compactFD'
       
       N = len(dom)
       # Master grid
       gridLG = [True, False, False, True, False]
       
       sdom = []
       cmats = []
       emats = []
       LE = []
       NEL = []
       
       # Loop over each element in dom (N-1 in 1D)
       for ee in range(N-1):
              # Get the element coordinates
              xa = dom[ee]
              xb = dom[ee+1]
              LE += [abs(xb - xa)]
              NEL += [NE]
              
              # Compute the local cubic-spline derivative matrix
              SDIMS1 = [xa, xb, LE[ee], NE, NE]
              SREFS = computeGrid(SDIMS1, *gridLG)
              sdom += [SREFS[1] + xa]
              DMS, LTR = computeLegendreDerivativeMatrix(SDIMS1)
              
              # Make a list of element matrices
              cmats += [LTR]
              emats += [DMS]
              
              if ee == 0:
                     edom = sdom[ee]
                     gdom = sdom[ee]
              else:
                     edom = np.append(edom, sdom[ee][0:])
                     gdom = np.append(gdom, sdom[ee][1:])
                     
       def getInterface(mat, ttr, ttc):
              #print(mat.shape)
              acol = np.delete(mat[:,ttc-1], ttr)
              bcol = np.delete(mat[:,ttc], ttr-1)
              arow = np.delete(mat[ttr-1,:], ttc)
              brow = np.delete(mat[ttr,:], ttc-1)
              
              return acol, bcol, arow, brow
       
       def getCommonDerivative2(xa, xb, da, db, ldex1, ldex2):

              La = abs(xa[-1] - xa[0])
              Lb = abs(xb[-1] - xb[0])
              
              # Scaling of matched coefficients here
              LM = (Lb + La)
              asc = La / LM
              bsc = Lb / LM
              
              dcom1 = np.append(da, np.zeros(ldex2[1]-1))
              dcom2 = np.append(np.zeros(ldex1[1]-1), db)
         
              # C0 continuity at the interface...
              DCOM = (asc * dcom1 + bsc * dcom2)
                            
              return DCOM
       
       def getCommonDerivative4(xa, xb, da, db, ldex1, ldex2):
              
              dom = np.append(xa, xb[1:]) - xa[-1]
              xdex = -len(dom)
              
              hp = abs(xb[1] - xb[0])
              hm = abs(xa[-2] - xa[-1])
              
              # Compute the stencil coefficients
              hr = (hm / hp)
              d = 0.25
              c = d * hr**4
              b = 0.5 * hr**3 * c + 2.5 * d 
              a = (2.0 * d - b) * hr**2 - 2.0 * c
              
              dcom1 = np.append(da, np.zeros(ldex2[1]-1))
              dcom2 = np.append(np.zeros(ldex1[1]-1), db)
              DCOM = np.zeros(abs(xdex))
              
              cdex = -ldex2[1]
              DCOM[cdex-1] = b
              DCOM[cdex] = -(a + b)
              DCOM[cdex+1] = a
              DCOM += (d * hm * dcom1 + \
                       c * hp * dcom2)
              DCOM *= (1.0 / (hp*(a+c) + hm*(d-b)))
              
              return DCOM
       
       def getCommonDerivative6(xa, xb, da, db, ldex1, ldex2):
              
              dom = np.append(xa, xb[1:]) - xa[-1]
              xdex = -len(dom)
              
              hp1 = abs(xb[1] - xb[0])
              hp2 = abs(xb[2] - xb[1])
              hm1 = abs(xa[-2] - xa[-1])
              hm2 = abs(xa[-3] - xa[-2])
              
              # 6TH ORDER CFD PATCHING EQUATION
              ND = 6
              CM6 = np.ones((ND,ND))
              c1 = hp1; c2 = hm1; c3 = hp2 + hp1; c4 = hm1 + hm2
              CM6[0,:] = [c1, -c2, +c3, -c4, -1.0, -1.0]
              CM6[1,:] = [c1**2, +c2**2, +c3**2, +c4**2, +2.0 * c2, -2.0 * c1]
              CM6[2,:] = [c1**3, -c2**3, +c3**3, -c4**3, -3.0 * c2**2, -3.0 * c1**2]
              CM6[3,:] = [c1**4, +c2**4, +c3**4, +c4**4, +4.0 * c2**3, -4.0 * c1**3]
              CM6[4,:] = [c1**5, -c2**5, +c3**5, -c4**5, -5.0 * c2**4, -5.0 * c1**4]
              CM6[5,:] = [c1**6, +c2**6, +c3**6, +c4**6, +6.0 * c2**5, -6.0 * c1**5]
              
              CMV = np.zeros(ND)
              CMV[0] = 1.0
              
              # Set end value to 1/3
              #alpha = 1.0 / 3.0
              #beta = 1.0 / 3.0
              #CMV -= (alpha * CM6[:,-2] + beta * CM6[:,-1])
              #CF = np.linalg.solve(CM6[0:-2,0:-2], CMV[0:-2])
              
              #beta = 1.0 / 3.0
              #CMV -= (beta * CM6[:,-1])
              #CF = np.linalg.solve(CM6[0:-1,0:-1], CMV[0:-1])
              #alpha = CF[4]
              
              CF = np.linalg.solve(CM6, CMV)
              alpha = CF[4]
              beta = CF[5]
              
              CFE = -np.sum(CF[0:ND-2])
              
              dcom1 = np.append(da, np.zeros(ldex2[1]-1))
              dcom2 = np.append(np.zeros(ldex1[1]-1), db)
              DCOM = np.zeros(abs(xdex))
              
              cdex = -ldex2[1]
              DCOM[cdex-2] = CF[3]
              DCOM[cdex-1] = CF[1]
              DCOM[cdex] = CFE
              DCOM[cdex+1] = CF[0]
              DCOM[cdex+2] = CF[2]
              DCOM -= (alpha * dcom1 + beta * dcom2)
              
              return DCOM

       # Loop over each dof and assemble the global matrix
       for ee in range(N-1):
                     
              tdexr = -(NEL[ee]+1)
              tdexc = -(NEL[ee]+1)
              
              if adjustedMethod:
                     if ee == 0:
                            GDMS = computeAdjustedOperatorNBC(emats[ee], emats[ee], -1)
                     elif ee == N-2:
                            DDMS = computeAdjustedOperatorNBC(emats[ee], emats[ee], 0)
                     else:
                            DDMS = computeAdjustedOperatorNBC_ends(emats[ee], emats[ee])
              else:
                     if ee == 0:
                            GDMS = emats[ee]
                     else:
                            DDMS = emats[ee]
              
              if ee > 0:
                     ldex1 = emats[ee-1].shape
                     ldex2 = emats[ee].shape
                                          
                     dom = np.append(sdom[ee-1], sdom[ee][1:]) - sdom[ee-1][-1]
                     ldex = -len(dom)
                     
                     if commonDerivativeMethod == 'interface_average':
                            da = emats[ee-1][-1,:]
                            db = emats[ee][0,:]
                            DCOM = getCommonDerivative2(sdom[ee-1], sdom[ee], da, db, ldex1, ldex2)
                     
                     if commonDerivativeMethod == 'order4_compactFD':
                            da = emats[ee-1][-2,:]
                            db = emats[ee][1,:]
                            DCOM = getCommonDerivative4(sdom[ee-1], sdom[ee], da, db, ldex1, ldex2)
                     
                     if commonDerivativeMethod == 'order6_compactFD':
                            da = emats[ee-1][-2,:]
                            db = emats[ee][1,:]
                            DCOM = getCommonDerivative6(sdom[ee-1], sdom[ee], da, db, ldex1, ldex2)
                     
                     # Repeat the center value and half
                     cdex = -ldex2[1]
                     center_value = 0.5 * DCOM[cdex]
                     DCOMR = np.insert(DCOM,cdex, center_value)
                     DCOMR[cdex] *= 0.5
                     
                     dcom1 = np.copy(DCOMR)
                     dcom2 = np.copy(DCOMR)
                     
                     # Append the new diagonal block
                     GDMS = scl.block_diag(GDMS, DDMS)
                     
                     if adjustedMethod:
                            if ee-1 == 0:
                                   rv = emats[ee-1][:,-1]
                                   s1b = 1.0 / emats[ee-1][-1,-1]
                                   E1b = np.outer(rv, s1b * dcom2)
                                   
                                   lv = emats[ee][:,0]
                                   rv = emats[ee][:,-1]
                                   R = emats[ee][0,0] * emats[ee][-1,-1] - emats[ee][0,-1] * emats[ee][-1,0] 
                                   
                                   E2a = np.outer(lv, emats[ee][-1,-1] * dcom1)
                                   E2a -= np.outer(rv, emats[ee][-1,0] * dcom1)
                                   E2a *= (1.0 / R)
                            elif ee == N-2:
                                   lv = emats[ee-1][:,0]
                                   rv = emats[ee-1][:,-1]
                                   R = emats[ee-1][0,0] * emats[ee-1][-1,-1] - emats[ee-1][0,-1] * emats[ee-1][-1,0] 
                                   
                                   E1b = np.outer(rv, emats[ee-1][0,0] * dcom2)
                                   E1b -= np.outer(lv, emats[ee-1][0,-1] * dcom2)
                                   E1b *= (1.0 / R)
                                   
                                   lv = emats[ee][:,0]
                                   s2a = 1.0 / emats[ee][0,0]
                                   E2a = np.outer(lv, s2a * dcom1)
                            else:
                                   lv = emats[ee-1][:,0]
                                   rv = emats[ee-1][:,-1]
                                   R = emats[ee-1][0,0] * emats[ee-1][-1,-1] - emats[ee-1][0,-1] * emats[ee-1][-1,0] 
                                   
                                   E1b = np.outer(rv, emats[ee-1][0,0] * dcom2)
                                   E1b -= np.outer(lv, emats[ee-1][0,-1] * dcom2)
                                   E1b *= (1.0 / R)
                                   
                                   lv = emats[ee][:,0]
                                   rv = emats[ee][:,-1]
                                   R = emats[ee][0,0] * emats[ee][-1,-1] - emats[ee][0,-1] * emats[ee][-1,0] 
                                   
                                   E2a = np.outer(lv, emats[ee][-1,-1] * dcom1)
                                   E2a -= np.outer(rv, emats[ee][-1,0] * dcom1)
                                   E2a *= (1.0 / R)
                                   
                            # update with off diagonal matrices (full adjustment)
                            GDMS[ldex-1:tdexr,ldex-1:] += E1b
                            GDMS[tdexr:,ldex-1:] += E2a
                            
                     #'''
                     # Get the rows/cols interface combine
                     acol, bcol, arow, brow = getInterface(GDMS, tdexr, tdexc)
                     mcol = acol + bcol
                     
                     # Delete redundant row/col
                     GDMS = np.delete(GDMS, tdexr, axis=0)
                     GDMS = np.delete(GDMS, tdexc, axis=1)
                     
                     # Set the merged row/col
                     GDMS[:,tdexc] = mcol
                     GDMS[tdexr,ldex:] = DCOM
                     #'''
       DDMSA = numericalCleanUp(GDMS)
       
       return DDMSA, gdom

# Compute Spectral Cubic Spline Element 1st derivative matrix
def computeSCSElementDerivativeMatrix(dom, NE, isLaguerreEnds):
       N = len(dom)
       
       EF0 = 1.0 
       EF1 = 1.0
       
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
                            DMB, CMB, scale = computeLaguerreDerivativeMatrix(SDIMS)
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
                            DMB, CMB, scale = computeLaguerreDerivativeMatrix(SDIMS)
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
              d = 0.25
              c = d * hr**4
              b = 1.0 / 8.0 * (5.0 + hr) 
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
       lg = np.amax(xi)
       b = lg / abs(ZH)
       
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
       
       return DDMC.astype(np.float64), STR_L, lg

def computeLegendreDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       xi, wlf = leglb(NZ)
       LT, DLT = legpolym(NZ, xi, True)
       
       b = 2.0 / ZH
       
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
       DDM = b * temp.dot(STR_L)
       
       DDMC = numericalCleanUp(DDM)
       
       #print(DDM[0,0], -NZ * (NZ + 1) / 2.0 / ZH)
       #print(DDM[-1,-1], NZ * (NZ + 1) / 2.0 / ZH)
       
       return DDMC, STR_L
       
       