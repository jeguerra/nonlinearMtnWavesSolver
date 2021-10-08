#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:59:22 2019

@author: TempestGuerra
"""

import scipy.linalg as scl
import scipy.sparse as sps
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
       
       DOP[tdex,:] = 0.0; DOP[:,tdex] = 0.0

       DOPC = numericalCleanUp(DOP)
       
       return DOPC

def computeAdjustedOperatorNBC_ends(D2A, DD):
       # D2A is the operator to adjust
       # DD is the 1st derivative operator
       
       R = (DD[0,0] * DD[-1,-1] - DD[0,-1] * DD[-1,0])
       
       V1 = (DD[-1,0] / R) * DD[:,-1] - (DD[-1,-1] / R) * DD[:,0] 
       DA1 = np.outer(V1, DD[0,:])
       
       V2 = (DD[0,-1] / R) * DD[:,0] - (DD[0,0] / R) * DD[:,-1] 
       DA2 = np.outer(V2, DD[-1,:])
       
       DOP = D2A + DA1 + DA2
       
       DOP[0,:] = 0.0; DOP[:,0] = 0.0
       DOP[-1,:] = 0.0; DOP[:,-1] = 0.0
       
       DOPC = numericalCleanUp(DOP)
       
       return DOPC

def computeAdjustedOperatorNBC_ends2(D2A, DD):
       # D2A is the operator to adjust
       # DD is the 1st derivative operator
       
       R = (DD[0,0] * DD[-1,-1] - DD[0,-1] * DD[-1,0])
       
       V1 = (DD[-1,0] / R) * DD[:,-1] - (DD[-1,-1] / R) * DD[:,0] 
       #DA1 = np.outer(V1, DD[0,:])
       #'''
       SM1 = sps.diags(DD[0,:])
       CM1 = V1
       for cc in range(1,SM1.shape[1]):
              CM1 = np.column_stack((CM1,V1))
       #'''
       V2 = (DD[0,-1] / R) * DD[:,0] - (DD[0,0] / R) * DD[:,-1] 
       #DA2 = np.outer(V2, DD[-1,:])
       #'''
       SM2 = sps.diags(DD[-1,:])
       CM2 = V2
       for cc in range(1,SM2.shape[1]):
              CM2 = np.column_stack((CM2,V2))
       
       DA1 = SM1.dot(CM1)
       DA2 = SM2.dot(CM2)
       #'''
       DOP = D2A + DA1 + DA2
       
       DOP[0,:] = 0.0; DOP[:,0] = 0.0
       DOP[-1,:] = 0.0; DOP[:,-1] = 0.0
       
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

# Compute Spectral Element 1st derivative matrix (2 sided coupling)
def computeSpectralElementDerivativeMatrix(dom, NE, adjustedMethod, nonCoincident):
              
       N = len(dom)
       # Master grid
       gridLG = [True, False, False, True, False]
       endGrid = [True, False, False, False, True]
       gridCH = [False, True, True, False, False]
       
       isLegendre = False
       isLaguerreEnd1 = False
       isLaguerreEnd2 = False
       nativeInterface = True
       CFDInterface = False
       ORDER = 6
       OFFSET = 1
       
       sdom = []
       dmats = []
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
              SDIMS = [xa, xb, LE[ee], NE, NE]
              
              if ee == 0:
                     if isLaguerreEnd1:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom += [-1.0 * np.flip(SREFS[1] + abs(xb))]
                            DMB, CMB, scale = computeLaguerreDerivativeMatrix(SDIMS)
                            RDMS = -1.0 * np.flip(np.flip(DMB,axis=0), axis=1)
                     else:
                            if not isLegendre:
                                   SREFS = computeGrid(SDIMS, *gridLG)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                            else:
                                   SREFS = computeGrid(SDIMS, *gridCH)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                                   
                     if adjustedMethod:
                            ADMS = np.copy(RDMS)
                            ADMS[:,-1] = 0.0
                     else:
                            ADMS = np.copy(RDMS)
                            
              elif ee == N-2:
                     if isLaguerreEnd2:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom += [SREFS[1] + xa]
                            DMB, CMB, scale = computeLaguerreDerivativeMatrix(SDIMS)
                            RDMS = 1.0 * DMB
                     else:
                            if not isLegendre:
                                   SREFS = computeGrid(SDIMS, *gridLG)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                            else:
                                   SREFS = computeGrid(SDIMS, *gridCH)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                     
                     if adjustedMethod:
                            ADMS = np.copy(RDMS)
                            ADMS[:,0] = 0.0
                     else:
                            ADMS = np.copy(RDMS)
              else:
                     # Compute the local derivative matrix
                     if isLegendre:
                            SREFS = computeGrid(SDIMS, *gridLG)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                     else:
                            SREFS = computeGrid(SDIMS, *gridCH)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                            
                     if adjustedMethod:
                            ADMS = np.copy(RDMS)
                            ADMS[:,0] = 0.0
                            ADMS[:,-1] = 0.0
                     else:
                            ADMS = np.copy(RDMS)
              
              # Make a list of element matrices
              dmats += [RDMS] # Raw derivative matrix
              emats += [ADMS] # Adjusted derivative matrix
              
              if ee == 0:
                     edom = sdom[ee]
                     gdom = sdom[ee]
              else:
                     edom = np.append(edom, sdom[ee][0:])
                     gdom = np.append(gdom, sdom[ee][1:])
       
       def getInterfaceDerivativeCFD(x, cdex):
              
              if OFFSET == 0:
                     ddm1 = computeCompactFiniteDiffDerivativeMatrix1(x, ORDER)
              else:
                     ddm1 = computeCompactFiniteDiffDerivativeMatrix1(x[OFFSET:-OFFSET], ORDER)
              
              cdex -= OFFSET
              
              df = 0.5
              di = ddm1[cdex,:]
              di[cdex] *= df
              dic = np.insert(di, cdex+1, di[cdex])
              
              padz = np.zeros(OFFSET)
              dic = np.concatenate((padz, dic, padz), axis=0)
              
              return dic

       # Loop over each dof and assemble the global matrix
       odexr = []
       odexc = []
       asc = []
       bsc = []
       csc = []
       GDMS = scl.block_diag(emats[0], emats[1])
       
       tdexr = -(NEL[1]+1)
       tdexc = -(NEL[1]+1)
       odexr += [GDMS.shape[0] + tdexr]
       odexc += [GDMS.shape[1] + tdexc]
       
       asc += [0.5]
       bsc += [LE[0]]
       csc += [LE[1]]
       
       if adjustedMethod:
              
              # One sided adjust on first block
              pad0 = np.zeros(NEL[0]+1)
              pad1 = np.zeros(NEL[1]+1)
              rv = np.copy(dmats[0][:,-1])
              
              dr = np.copy(dmats[0][-1,:]); dr[-1] = 0.0
              dr = np.concatenate((dr, pad1))
              
              #pr = np.copy(dmats[1][0,:])
              #pr = np.concatenate((pad0, pr))
              
              LM = (LE[0] + LE[1])
              pr = np.concatenate((bsc[-1] / LM * dmats[0][-1,:], \
                                   csc[-1] / LM * dmats[1][0,:]))
              
              qc1 = (1.0 / dmats[0][-1,-1]) * (pr - dr)
              
              Er = np.outer(rv, qc1)
              GDMS[0:NEL[0]+1,:] += Er
              
              for ee in range(2,N-1):
                     ii = ee - 1
                     tdexr = -(NEL[ee]+1)
                     tdexc = -(NEL[ee]+1)
                     
                     # Append the next diagonal block
                     GDMS = scl.block_diag(GDMS, emats[ee])
                     
                     odexr += [GDMS.shape[0] + tdexr]
                     odexc += [GDMS.shape[1] + tdexc]
                     
                     # Scaling of matched coefficients here
                     LM1 = (LE[ii-1] + LE[ii])
                     LM2 = (LE[ii] + LE[ii+1])
                     asc += [LE[ii-1]]
                     bsc += [LE[ii]]
                     csc += [LE[ii+1]]
                     
                     pad0 = np.zeros(NEL[ii-1]+1)
                     pad1 = np.zeros(NEL[ii]+1)
                     pad2 = np.zeros(NEL[ii+1]+1)
                     
                     # Inter-element common interface derivatives (left and right)
                     #pl = np.copy(dmats[ii-1][-1,:])
                     #pl = np.concatenate((pl, pad1, pad2))
                     pl = np.concatenate((asc[-1] / LM1 * dmats[ii-1][-1,:], \
                                          bsc[-1] / LM1 * dmats[ii][0,:], pad2))
                     
                     #pr = np.copy(dmats[ii+1][0,:])
                     #pr = np.concatenate((pad0, pad1, pr))
                     pr = np.concatenate((pad0, bsc[-1] / LM2 * dmats[ii][-1,:], \
                                          csc[-1] / LM2 * dmats[ii+1][0,:]))
                     
                     # Intra-element native interface derivatives
                     dl = np.copy(dmats[ii][0,:]); dl[0] = 0.0; dl[-1] = 0.0
                     dl = np.concatenate((pad0, dl, pad2))
                     
                     dr = np.copy(dmats[ii][-1,:]); dr[0] = 0.0; dr[-1] = 0.0
                     dr = np.concatenate((pad0, dr, pad2))
                     
                     # Scale factor
                     R = dmats[ii][0,0] * dmats[ii][-1,-1] - dmats[ii][0,-1] * dmats[ii][-1,0]
                     
                     # Right adjust
                     rv = np.copy(dmats[ii][:,-1])                            
                     qc1 = (dmats[ii][0,0] / R) * (pr - dr) - \
                            (dmats[ii][-1,0] / R) * (pl - dl)
                     E1 = np.outer(rv, qc1)
                     
                     # Left adjust 
                     lv = np.copy(dmats[ii][:,0])                            
                     qc2 = (dmats[ii][-1,-1] / R) * (pl - dl) - \
                            (dmats[ii][0,-1] / R) * (pr - dr)
                     E2 = np.outer(lv, qc2)
                     
                     bdex = NEL[ii+1] + 1
                     tdex = (NEL[ii] + NEL[ii+1]) + 2
                     ldex = (NEL[ii-1] + NEL[ii] + NEL[ii+1]) + 3
                     GDMS[-tdex:-bdex,-ldex:] += E1
                     GDMS[-tdex:-bdex,-ldex:] += E2
              
              # One sided adjust on last block
              pad0 = np.zeros(NEL[-2]+1)
              pad1 = np.zeros(NEL[-1]+1)
              lv = np.copy(dmats[-1][:,0])
              
              dl = np.copy(dmats[-1][0,:]); dl[0] = 0.0
              dl = np.concatenate((pad0, dl))
              
              #pl = np.copy(dmats[-2][-1,:])
              #pl = np.concatenate((pl, pad1))
              
              LM = (LE[-2] + LE[-1])
              pl = np.concatenate((bsc[-1] / LM * dmats[-2][-1,:], \
                                   csc[-1] / LM * dmats[-1][0,:]))
              
              qc = (1.0 / dmats[-1][0,0]) * (pl - dl)
              
              El = np.outer(lv, qc)
              ldex = (NEL[-2] + NEL[-1]) + 2
              GDMS[-(NEL[-1]+1):,-ldex:] += El
       else:
              for ee in range(2,N-1):
                     # Append the next diagonal block
                     GDMS = scl.block_diag(GDMS, emats[ee])
                     
                     odexr += [GDMS.shape[0] + tdexr]
                     odexc += [GDMS.shape[1] + tdexc]
                     
                     # Scaling of matched coefficients here
                     LM = (LE[ee-1] + LE[ee])
                     asc += [LE[ee-1] / LM]
                     bsc += [LE[ee] / LM]
                     
       if nonCoincident:
              # Linear combination of columns
              for ii in range(len(odexr)):
                     cdex = odexc[ii]
                     
                     # merge rows and columns
                     GDMS[:,cdex-1] += GDMS[:,cdex]
                     GDMS[:,cdex] = 0.0
                     
              GDMS = np.delete(GDMS, odexc, axis=1)
              domain = gdom
              
              if nativeInterface:
                     # Delete redundant row/col
                     GDMS1 = 0.5 * np.copy(GDMS)
                     GDMS2 = 0.5 * np.copy(GDMS)
                     #GDMS1[np.array(odexr)-1,:] *= 2.0 * np.expand_dims(bsc, axis=1)
                     #GDMS2[np.array(odexr),:] *= 2.0 * np.expand_dims(asc, axis=1)
                     GDMS1 = np.delete(GDMS1, np.array(odexr), axis=0)
                     GDMS2 = np.delete(GDMS2, np.array(odexr)-1, axis=0)              
                     GDMS = GDMS1 + GDMS2
              else:
                     GDMS = np.delete(GDMS, np.array(odexr), axis=0)
       else:
              domain = edom
                     
       DDMSA = numericalCleanUp(GDMS)
       
       return DDMSA, domain

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
def computeCompactFiniteDiffDerivativeMatrix1(dom, order):
       # Initialize the left and right derivative matrices
       N = len(dom)
       LDM = np.zeros((N,N)) # tridiagonal
       RDM = np.zeros((N,N)) # centered difference
       
       if order == 4:
              '''
              # Loop over each interior point in the irregular grid
              for ii in range(1,N-1):
                     # Get the metric weights
                     hp = abs(dom[ii+1] - dom[ii])
                     hm = abs(dom[ii] - dom[ii-1])
                     
                     # Compute the stencil coefficients
                     hr = (hm / hp)
                     d = 0.25
                     c = d * hr**4
                     b = 0.5 * hr**3 * c + 2.5 * d 
                     a = (2.0 * d - b) * hr**2 - 2.0 * c
                     
                     # Write the right equation
                     RDM[ii,ii-1] = b
                     RDM[ii,ii] = -(a + b)
                     RDM[ii,ii+1] = a
                     # Write the left equation
                     LDM[ii,ii-1] = -d * hm
                     LDM[ii,ii] = (hp * (a + c) + hm * (d - b))
                     LDM[ii,ii+1] = -c * hp
              '''       
              # Loop over each interior point in the irregular grid
              for ii in range(1,N-1):
                     # Get the metric weights
                     hp1 = abs(dom[ii+1] - dom[ii])
                     hm1 = abs(dom[ii] - dom[ii-1])
                     
                     # Compute the stencil coefficients
                     ND = 4
                     CM4 = np.ones((ND,ND))
                     c1 = hp1; c2 = hm1
                     CM4[0,:] = [c1, -c2, -1.0, -1.0]
                     CM4[1,:] = [c1**2, +c2**2, +2.0 * c2, -2.0 * c1]
                     CM4[2,:] = [c1**5, -c2**5, -5.0 * c2**4, -5.0 * c1**4]
                     CM4[3,:] = [c1**6, +c2**6, +6.0 * c2**5, -6.0 * c1**5]
                     
                     CMV = np.zeros(ND)
                     CMV[0] = 1.0
                     
                     '''
                     CF = np.linalg.solve(CM4, CMV)
                     alpha = CF[2]
                     beta = CF[3]
                     CFE = -np.sum(CF[0:ND-2])
                     '''
                     # Constraint alpha = beta
                     CM4[:,-2] += CM4[:,-1]
                     CF = np.linalg.solve(CM4[0:-1,0:-1], CMV[0:-1])
                     alpha = CF[2]
                     beta = CF[2]
                     CFE = -np.sum(CF[0:ND-2])
                     
                     # Write the right equation
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     
                     # Write the left equation
                     LDM[ii,ii-1] = alpha
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = beta
              
       if order == 6:
              # Loop over each interior point in the irregular grid
              for ii in [1, N-2]:
                     # Get the metric weights
                     hp1 = abs(dom[ii+1] - dom[ii])
                     hm1 = abs(dom[ii] - dom[ii-1])
                     
                     # Compute the stencil coefficients
                     ND = 4
                     CM4 = np.ones((ND,ND))
                     c1 = hp1; c2 = hm1
                     CM4[0,:] = [c1, -c2, -1.0, -1.0]
                     CM4[1,:] = [c1**2, +c2**2, +2.0 * c2, -2.0 * c1]
                     CM4[2,:] = [c1**5, -c2**5, -5.0 * c2**4, -5.0 * c1**4]
                     CM4[3,:] = [c1**6, +c2**6, +6.0 * c2**5, -6.0 * c1**5]
                     
                     CMV = np.zeros(ND)
                     CMV[0] = 1.0
                     '''
                     CF = np.linalg.solve(CM4, CMV)
                     alpha = CF[2]
                     beta = CF[3]
                     CFE = -np.sum(CF[0:ND-2])
                     '''
                     # Constraint alpha = beta
                     CM4[:,-2] += CM4[:,-1]
                     CF = np.linalg.solve(CM4[0:-1,0:-1], CMV[0:-1])
                     alpha = CF[2]
                     beta = CF[2]
                     CFE = -np.sum(CF[0:ND-2])
                     
                     # Write the right equation
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     
                     # Write the left equation
                     LDM[ii,ii-1] = alpha
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = beta
              
              # Loop over each interior point in the irregular grid
              for ii in range(2,N-2):
                     # Get the metric weights
                     hp1 = abs(dom[ii+1] - dom[ii])
                     hm1 = abs(dom[ii] - dom[ii-1])
                     hp2 = abs(dom[ii+2] - dom[ii+1])
                     hm2 = abs(dom[ii-1] - dom[ii-2])
                     
                     # Compute the stencil coefficients
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
                     
                     # Constraint alpha = beta
                     CM6[:,-2] += CM6[:,-1]
                     CF = np.linalg.solve(CM6[0:-1,0:-1], CMV[0:-1])
                     alpha = CF[4]
                     beta = CF[4]
                     CFE = -np.sum(CF[0:ND-2])
                     
                     # Write the right equation
                     RDM[ii,ii-2] = CF[3]
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     RDM[ii,ii+2] = CF[2]
                     
                     # Write the left equation
                     LDM[ii,ii-1] = alpha
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = beta
                     
       if order == 8:
              # Loop over each interior point in the irregular grid
              for ii in [1,N-2]:
                     # Get the metric weights
                     hp1 = abs(dom[ii+1] - dom[ii])
                     hm1 = abs(dom[ii] - dom[ii-1])
                     
                     # Compute the stencil coefficients
                     ND = 4
                     CM4 = np.ones((ND,ND))
                     c1 = hp1; c2 = hm1
                     CM4[0,:] = [c1, -c2, -1.0, -1.0]
                     CM4[1,:] = [c1**2, +c2**2, +2.0 * c2, -2.0 * c1]
                     CM4[2,:] = [c1**5, -c2**5, -5.0 * c2**4, -5.0 * c1**4]
                     CM4[3,:] = [c1**6, +c2**6, +6.0 * c2**5, -6.0 * c1**5]
                     
                     CMV = np.zeros(ND)
                     CMV[0] = 1.0
                     
                     '''
                     CF = np.linalg.solve(CM4, CMV)
                     alpha = CF[2]
                     beta = CF[3]
                     CFE = -np.sum(CF[0:ND-2])
                     '''
                     # Constraint alpha = beta
                     CM4[:,-2] += CM4[:,-1]
                     CF = np.linalg.solve(CM4[0:-1,0:-1], CMV[0:-1])
                     alpha = CF[2]
                     beta = CF[2]
                     CFE = -np.sum(CF[0:ND-2])
                     
                     # Write the right equation
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     
                     # Write the left equation
                     LDM[ii,ii-1] = alpha
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = beta
              
              # Loop over each interior point in the irregular grid
              for ii in [2,N-3]:
                     # Get the metric weights
                     hp1 = abs(dom[ii+1] - dom[ii])
                     hm1 = abs(dom[ii] - dom[ii-1])
                     hp2 = abs(dom[ii+2] - dom[ii+1])
                     hm2 = abs(dom[ii-1] - dom[ii-2])
                     
                     # Compute the stencil coefficients
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
                     '''
                     CF = np.linalg.solve(CM6, CMV)
                     alpha = CF[4]
                     beta = CF[5]
                     '''
                     CM6[:,-2] += CM6[:,-1]
                     CF = np.linalg.solve(CM6[0:-1,0:-1], CMV[0:-1])
                     alpha = CF[4]
                     beta = CF[4]
              
                     CFE = -np.sum(CF[0:ND-2])
                     
                     # Write the right equation
                     RDM[ii,ii-2] = CF[3]
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     RDM[ii,ii+2] = CF[2]
                     
                     # Write the left equation
                     LDM[ii,ii-1] = alpha
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = beta
                     
              # Loop over each interior point in the irregular grid
              for ii in range(3,N-3):
                     # Get the metric weights
                     hp1 = abs(dom[ii+1] - dom[ii])
                     hm1 = abs(dom[ii] - dom[ii-1])
                     hp2 = abs(dom[ii+2] - dom[ii+1])
                     hm2 = abs(dom[ii-1] - dom[ii-2])
                     hp3 = abs(dom[ii+3] - dom[ii+2])
                     hm3 = abs(dom[ii-2] - dom[ii-3])
                     
                     # Compute the stencil coefficients
                     ND = 10
                     CM8 = np.ones((ND,ND))
                     c1 = hp1; c2 = hm1
                     c3 = hp2 + hp1; c4 = hm1 + hm2
                     c5 = hp3 + hp2 + hp1; c6 = hm1 + hm2 + hm3
                     ''' Tridiagonal left, Septadiagonal right
                     CM8[0,:] = [c1, -c2, +c3, -c4, c5, -c6, -1.0, -1.0]
                     CM8[1,:] = [c1**2, +c2**2, +c3**2, +c4**2, +c5**2, +c6**2, +2.0 * c2, -2.0 * c1]
                     CM8[2,:] = [c1**3, -c2**3, +c3**3, -c4**3, +c5**3, -c6**3, -3.0 * c2**2, -3.0 * c1**2]
                     CM8[3,:] = [c1**4, +c2**4, +c3**4, +c4**4, +c5**4, +c6**4, +4.0 * c2**3, -4.0 * c1**3]
                     CM8[4,:] = [c1**5, -c2**5, +c3**5, -c4**5, +c5**5, -c6**5, -5.0 * c2**4, -5.0 * c1**4]
                     CM8[5,:] = [c1**6, +c2**6, +c3**6, +c4**6, +c5**6, +c6**6, +6.0 * c2**5, -6.0 * c1**5]
                     CM8[6,:] = [c1**7, -c2**7, +c3**7, -c4**7, +c5**7, -c6**7, -7.0 * c2**6, -7.0 * c1**6]
                     CM8[7,:] = [c1**8, +c2**8, +c3**8, +c4**8, +c5**8, +c6**8, +8.0 * c2**7, -8.0 * c1**7]
                     '''
                     #''' Pentadiagonal left, Septadiagonal right
                     CM8[0,:] = [c1, -c2, +c3, -c4, c5, -c6, -1.0, -1.0, -1.0, -1.0]
                     CM8[1,:] = [c1**2, +c2**2, +c3**2, +c4**2, +c5**2, +c6**2, +2.0 * c2, -2.0 * c1, +2.0 * c4, -2.0 * c3]
                     CM8[2,:] = [c1**3, -c2**3, +c3**3, -c4**3, +c5**3, -c6**3, -3.0 * c2**2, -3.0 * c1**2, -3.0 * c4**2, -3.0 * c3**2]
                     CM8[3,:] = [c1**4, +c2**4, +c3**4, +c4**4, +c5**4, +c6**4, +4.0 * c2**3, -4.0 * c1**3, +4.0 * c4**3, -4.0 * c3**3]
                     CM8[4,:] = [c1**5, -c2**5, +c3**5, -c4**5, +c5**5, -c6**5, -5.0 * c2**4, -5.0 * c1**4, -5.0 * c4**4, -5.0 * c3**4]
                     CM8[5,:] = [c1**6, +c2**6, +c3**6, +c4**6, +c5**6, +c6**6, +6.0 * c2**5, -6.0 * c1**5, +6.0 * c4**5, -6.0 * c3**5]
                     CM8[6,:] = [c1**7, -c2**7, +c3**7, -c4**7, +c5**7, -c6**7, -7.0 * c2**6, -7.0 * c1**6, -7.0 * c4**6, -7.0 * c3**6]
                     CM8[7,:] = [c1**8, +c2**8, +c3**8, +c4**8, +c5**8, +c6**8, +8.0 * c2**7, -8.0 * c1**7, +8.0 * c4**7, -8.0 * c3**7]
                     CM8[8,:] = [c1**9, -c2**9, +c3**9, -c4**9, +c5**9, -c6**9, -9.0 * c2**8, -9.0 * c1**8, -9.0 * c4**8, -9.0 * c3**8]
                     CM8[9,:] = [c1**10, +c2**10, +c3**10, +c4**10, +c5**10, +c6**10, +10.0 * c2**9, -10.0 * c1**9, +10.0 * c4**9, -10.0 * c3**9]
                     #'''
                     CMV = np.zeros(ND)
                     CMV[0] = 1.0
                     '''
                     CF = np.linalg.solve(CM8, CMV)
                     alpha = CF[4]
                     beta = CF[5]
                     theta = CF[6]
                     rho = CF[7]
                     '''
                     CM8[:,-2] += CM8[:,-1]
                     CM8[:,-4] += CM8[:,-3]
                     sdex = np.array([0, 1, 2, 3, 4, 5, 6, 8])
                     CF = np.linalg.solve(CM8[np.ix_(sdex,sdex)], CMV[sdex])
                     alpha = CF[6]
                     beta = CF[6]
                     theta = CF[7]
                     rho = CF[7]
              
                     CFE = -np.sum(CF[0:ND-4])
                     
                     # Write the right equation
                     RDM[ii,ii-3] = CF[5]
                     RDM[ii,ii-2] = CF[3]
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     RDM[ii,ii+2] = CF[2]
                     RDM[ii,ii+3] = CF[4]
                     
                     # Write the left equation
                     LDM[ii,ii-2] = theta
                     LDM[ii,ii-1] = alpha
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = beta
                     LDM[ii,ii+2] = rho
       
       '''
       # Coefficients for 4th order compact one-sided schemes
       hp1 = dom[1] - dom[0]
       hm1 = dom[-2] - dom[-1]
       hp2 = dom[2] - dom[1]
       hm2 = dom[-3] - dom[-2]
       hp3 = dom[3] - dom[2]
       hm3 = dom[-4] - dom[-3]

       # Compute the stencil coefficients
       ND = 8
       CM8 = np.ones((ND,ND))
       c1 = hp1; c2 = hm1
       c3 = hp2 + hp1; c4 = hm1 + hm2
       c5 = hp3 + hp2 + hp1; c6 = hm1 + hm2 + hm3
       CM8[0,:] = [c1, -c2, +c3, -c4, c5, -c6, -1.0, -1.0]
       CM8[1,:] = [c1**2, +c2**2, +c3**2, +c4**2, +c5**2, +c6**2, +2.0 * c2, -2.0 * c1]
       CM8[2,:] = [c1**3, -c2**3, +c3**3, -c4**3, +c5**3, -c6**3, -3.0 * c2**2, -3.0 * c1**2]
       CM8[3,:] = [c1**4, +c2**4, +c3**4, +c4**4, +c5**4, +c6**4, +4.0 * c2**3, -4.0 * c1**3]
       CM8[4,:] = [c1**5, -c2**5, +c3**5, -c4**5, +c5**5, -c6**5, -5.0 * c2**4, -5.0 * c1**4]
       CM8[5,:] = [c1**6, +c2**6, +c3**6, +c4**6, +c5**6, +c6**6, +6.0 * c2**5, -6.0 * c1**5]
       CM8[6,:] = [c1**7, -c2**7, +c3**7, -c4**7, +c5**7, -c6**7, -7.0 * c2**6, -7.0 * c1**6]
       CM8[7,:] = [c1**8, +c2**8, +c3**8, +c4**8, +c5**8, +c6**8, +8.0 * c2**7, -8.0 * c1**7]
       
       CMV = np.zeros(ND)
       CMV[0] = 1.0
       
       ford = [1, 3, 5, -2]
       CMF = np.delete(CM8, ford, axis=0)
       CMF = np.delete(CMF, ford, axis=1)
       
       CMF_V = np.delete(CMV, ford, axis=0)
       
       CF_F = np.linalg.solve(CMF, CMF_V)
       beta = CF_F[-1]
       '''
       '''
       LDM[0,0] = 1.0
       LDM[0,1] = beta
       RDM[0,0] = -np.sum(CF_F[0:-1])
       RDM[0,1] = CF_F[0]
       RDM[0,2] = CF_F[1]
       RDM[0,3] = CF_F[2]
       print(LDM[0,1], RDM[0,0], (RDM[0,1] + RDM[0,2] + RDM[0,3]))
       #'''
       '''
       LDM[N-1,N-1] = 1.0
       LDM[N-1,N-2] = beta
       RDM[N-1,N-1] = np.sum(CF_F[0:-1])
       RDM[N-1,N-2] = -CF_F[0]
       RDM[N-1,N-3] = -CF_F[1]
       RDM[N-1,N-4] = -CF_F[2]
       '''
       #'''
       hp2 = dom[1] - dom[0]
       hp3 = dom[2] - dom[1]
       LDM[0,0] = 1.0
       LDM[0,1] = (hp2 + hp3) / hp3
       RDM[0,0] = -(3.0 * hp2 + 2.0 * hp3) / (hp2 * (hp2 + hp3))
       RDM[0,1] = (hp2 + hp3) * (2.0 * hp3 - hp2) / (hp2 * hp3**2)
       RDM[0,2] = (hp2**2) / (hp3**2 * (hp2 + hp3))
       #'''
       #'''
       hp2 = dom[N-2] - dom[N-1]
       hp3 = dom[N-3] - dom[N-2]
       LDM[N-1,N-1] = 1.0
       LDM[N-1,N-2] = (hp2 + hp3) / hp3
       RDM[N-1,N-1] = -(3.0 * hp2 + 2.0 * hp3) / (hp2 * (hp2 + hp3))
       RDM[N-1,N-2] = (hp2 + hp3) * (2.0 * hp3 - hp2) / (hp2 * hp3**2)
       RDM[N-1,N-3] = (hp2**2) / (hp3**2 * (hp2 + hp3))
       #'''
       # Get the derivative matrix
       DDM = np.linalg.solve(LDM, RDM)
       
       DDMA = numericalCleanUp(DDM)
       
       return DDMA

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
       
       