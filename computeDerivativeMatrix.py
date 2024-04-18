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

def computeAdjustedOperatorNBC(D2A, DD, tdex):
       # D2A is the operator to adjust
       # DD is the 1st derivative operator
       R = DD[tdex,tdex]
       dr = DD[tdex,:]
       dc = DD[:,tdex]
       
       DOP = np.copy(D2A)   
       DA = 1.0 / R * np.expand_dims(dc, axis=1) @ np.expand_dims(dr, axis=0)
       DOP -= DA

       DOP[tdex,:] = 0.0
       DOP[:,tdex] = 0.0
       
       return DOP

def computeAdjustedOperatorNBC_ends(D2A, DD):
       # D2A is the operator to adjust
       # DD is the 1st derivative operator
       
       R = (DD[0,0] * DD[-1,-1] - DD[0,-1] * DD[-1,0])
       
       lv = DD[0,:]
       rv = DD[-1,:]
       
       V1 = (DD[-1,-1] / R) * (-lv) + (DD[0,-1] / R) * rv 
       DA1 = np.outer(DD[:,0], V1)
       
       V2 = (DD[-1,0] / R) * lv - (DD[0,0] / R) * (-rv) 
       DA2 = np.outer(DD[:,-1], V2)
       
       DOP = D2A + DA1 + DA2
       
       DOP[0,:] = 0.0; DOP[:,0] = 0.0
       DOP[-1,:] = 0.0; DOP[:,-1] = 0.0
              
       return DOP

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

def numericalCleanUp(DDM):
       
       N = DDM.shape
       ZTOL = 1.0E-16
       DDMC = np.copy(DDM)
       # Clean up numerical zeros
       for ii in range(N[0]):
              for jj in range(N[1]):
                     if abs(DDM[ii,jj]) < ZTOL:
                            DDMC[ii,jj] = 0.0
       return DDMC

def removeLeastSingularValue(DDM):
       mr = 1
       nm = DDM.shape
       U, s, Vh = scl.svd(DDM)
       S = scl.diagsvd(s[0:-mr], nm[0]-mr, nm[1]-mr)
       DDM = U[:,0:-mr].dot(S)
       DDM = DDM.dot(Vh[0:-mr,:])
       
       DDM = numericalCleanUp(DDM)
       
       return DDM

# Compute Spectral Element 1st derivative matrix (2 sided coupling)
def computeSpectralElementDerivativeMatrix5E(dom, NE, nonCoincident, ORDER):
       
       N = len(dom)
       
       if (N-1) != 5:
              print('@@ 5 ELEMENT C1 ONLY! @@')
              return
       
       # Master grid
       gridLG = [True, False, False, True, False]
       endGrid = [True, False, False, False, True]
       gridCH = [False, True, True, False, False]
       
       isLegendre = False
       isCSEInterface = False
       isCFDInterface = True
       OFFSET = 0
       
       sdom = []
       dmats = []
       LE = []
       NEL = []
       
       def function2(x, L):
       
              A = 4.0
              B = 2.0
              C = 4.0
              Y = C * np.exp(-A / L * x) * np.sin(B * mt.pi / L * x)
              DY = -(A * C) / L * np.exp(-A / L * x) * np.sin(B * mt.pi / L * x)
              DY += (B * C) * mt.pi / L * np.exp(-A / L * x) * np.cos(B * mt.pi / L * x)
              
              return Y, DY
       
       def getInterfaceDerivativeCSE(x, cdex):
              if OFFSET == 0:
                     ddm0 = computeCompactFiniteDiffDerivativeMatrix1(x, ORDER)
                     ddm1, d2 = computeCubicSplineDerivativeMatrix(x, True, False, False, False, ddm0)
              else:
                     xo = x[OFFSET:-OFFSET]
                     ddm0 = computeCompactFiniteDiffDerivativeMatrix1(xo, ORDER)
                     ddm1, d2 = computeCubicSplineDerivativeMatrix(xo, True, False, False, False, ddm0)
              
              cdex -= OFFSET
              
              df = 0.5
              di = ddm1[cdex,:]
              di[cdex] *= df
              dic = np.insert(di, cdex+1, di[cdex])
              
              padz = np.zeros(OFFSET)
              dic = np.concatenate((padz, dic, padz), axis=0)
              
              return dic
              
              return dic
       
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
       
       # Loop over each element in dom (N-1 in 1D)
       for ee in range(N-1):
              # Get the element coordinates
              xa = dom[ee]
              xb = dom[ee+1]
              LE += [abs(xb - xa)]
              NEL += [NE]
              SDIMS = [xa, xb, LE[ee], NE, NE]
              
              if ee == 0:
                     if isLegendre:
                            SREFS = computeGrid(SDIMS, *gridLG)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                     else:
                            SREFS = computeGrid(SDIMS, *gridCH)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                            #RDMS = computeCompactFiniteDiffDerivativeMatrix1(sdom[-1], ORDER)
                            #RDMS, d2 = computeCubicSplineDerivativeMatrix(sdom[-1], True, False, False, False, ADMS)
                            
              elif ee == N-2:
                     if isLegendre:
                            SREFS = computeGrid(SDIMS, *gridLG)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                     else:
                            SREFS = computeGrid(SDIMS, *gridCH)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                            #RDMS = computeCompactFiniteDiffDerivativeMatrix1(sdom[-1], ORDER)
                            #RDMS, d2 = computeCubicSplineDerivativeMatrix(sdom[-1], True, False, False, False, ADMS)
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
                            #RDMS = computeCompactFiniteDiffDerivativeMatrix1(sdom[-1], ORDER)
                                          
              # Make a list of element matrices
              dmats += [RDMS] # Raw derivative matrix
              
              if ee == 0:
                     edom = sdom[ee]
                     gdom = sdom[ee]
              else:
                     edom = np.append(edom, sdom[ee][0:])
                     gdom = np.append(gdom, sdom[ee][1:])
                     
      # Assemble the raw matrix
       odexr = [0]
       odexc = [0]
       GDMS = np.copy(dmats[0])
       for ee in range(1,N-1):
              # Append the next diagonal block
              GDMS = scl.block_diag(GDMS, dmats[ee])
              
              # Get the indices of coincident interfaces (left)
              tdexr = -(NEL[ee]+1)
              tdexc = -(NEL[ee]+1)
              odexr += [GDMS.shape[0] + tdexr]
              odexc += [GDMS.shape[1] + tdexc]
              
       odexr += [GDMS.shape[0] - 1]
       odexc += [GDMS.shape[1] - 1]
       
       # Compute the continuous common derivatives
       dr = [0, 0, 0, 0, 0]
       dl = [0, 0, 0, 0, 0]
       pr = [0, 0, 0, 0, 0]
       pl = [0, 0, 0, 0, 0]
       
       # Get intra-element interface derivatives for this element (dl and dr)
       for ee in range(N-1):
              dl[ee] = np.copy(dmats[ee][0,:])
              dr[ee] = np.copy(dmats[ee][-1,:])
              
              if ee == 0:
                     
                     dr[ee][-1] = 0.0
                     pad_right = np.zeros(GDMS.shape[1] - len(dr[ee]))
                     dr[ee] = np.concatenate((dr[ee], pad_right))
                     
              elif ee == N-2:
                     
                     dl[ee][0] = 0.0
                     pad_left = np.zeros(GDMS.shape[1] - len(dl[ee]))
                     dl[ee] = np.concatenate((pad_left, dl[ee]))
                     
              else:
                     
                     dl[ee][0] = 0.0; dl[ee][-1] = 0.0
                     pad_left = np.zeros(odexc[ee])
                     pad_right = np.zeros(GDMS.shape[1] - len(pad_left) - len(dl[ee]))
                     dl[ee] = np.concatenate((pad_left, dl[ee], pad_right))
                     
                     dr[ee][0] = 0.0; dr[ee][-1] = 0.0
                     pad_left = np.zeros(odexc[ee])
                     pad_right = np.zeros(GDMS.shape[1] - len(pad_left) - len(dr[ee]))
                     dr[ee] = np.concatenate((pad_left, dr[ee], pad_right))
       
       # Set the interior interface of the boundary elements
       pl[0] = np.zeros(1)
       pr[-1] = np.zeros(1)
       
       if isCFDInterface:
              pr[0] = getInterfaceDerivativeCFD(np.concatenate((sdom[0], sdom[1][1:])), NEL[1])
              pl[-1] = getInterfaceDerivativeCFD(np.concatenate((sdom[-2], sdom[-1][1:])), NEL[-1])
       
       if isCSEInterface:
              pr[0] = getInterfaceDerivativeCSE(np.concatenate((sdom[0], sdom[1][1:])), NEL[1])
              pl[-1] = getInterfaceDerivativeCSE(np.concatenate((sdom[-2], sdom[-1][1:])), NEL[-1])
                     
       pad_right = np.zeros(GDMS.shape[1] - len(pr[0]))
       pr[0] = np.concatenate((pr[0], pad_right))
       
       pad_left = np.zeros(GDMS.shape[1] - len(pl[-1]))
       pl[-1] = np.concatenate((pad_left, pl[-1]))
       
       # Set the interior interface of the first interior elements
       pl[1] = np.copy(pr[0])
       pr[-2] = np.copy(pl[-1])
       
       EE = 2
       
       # Solve the middle element EE = 2
       R1 = dmats[EE-1][0,0] * dmats[EE-1][-1,-1] - dmats[EE-1][0,-1] * dmats[EE-1][-1,0]
       R2 = dmats[EE][0,0] * dmats[EE][-1,-1] - dmats[EE][0,-1] * dmats[EE][-1,0]
       R3 = dmats[EE+1][0,0] * dmats[EE+1][-1,-1] - dmats[EE+1][0,-1] * dmats[EE+1][-1,0]
       
       # Right hand side 1
       RHS1 = -(dmats[EE-1][-1,0] / R1) * (pr[0] - dl[EE-1]) - (dmats[EE-1][0,0] / R1) * dr[EE-1] 
       RHS1 += (dmats[EE][-1,-1] / R2) * dl[EE] - (dmats[EE][0,-1] / R2) * dr[EE] 
       
       # Right hand side 2
       RHS2 = -(dmats[EE+1][0,-1] / R3) * (pl[-1] - dr[EE+1]) - (dmats[EE+1][-1,-1] / R3) * dl[EE+1]
       RHS2 += (dmats[EE][0,0] / R2) * dr[EE] - (dmats[EE][-1,0] / R2) * dl[EE]
       
       RHS = np.vstack((np.expand_dims(RHS1, axis=0), np.expand_dims(RHS2, axis=0)))
       
       AM = np.zeros((2,2))
       AM[0,0] = (dmats[EE][-1,-1] / R2) - (dmats[EE-1][0,0] / R1)
       AM[0,1] = -(dmats[EE][0,-1] / R2)
       AM[1,0] = -(dmats[EE][-1,0] / R2)
       AM[1,1] = (dmats[EE][0,-0] / R2) - (dmats[EE+1][-1,-1] / R3)
       
       # Solve for the interior common derivatives
       PDI = np.linalg.solve(AM, RHS)
       pl[EE] = PDI[0,:]
       pr[EE] = PDI[1,:]
       
       pr[1] = np.copy(pl[EE])
       pl[-2] = np.copy(pr[EE])
       
       # Go over each element and couple interfaces
       for ee in range(N-1):
              # Set common derivatives
              if ee > 0 and ee < N-2:
                     GDMS[odexr[ee]-1,:] = pl[ee]
                     GDMS[odexr[ee],:] = pl[ee]
                     GDMS[odexr[ee+1]-1,:] = pr[ee]
                     GDMS[odexr[ee+1],:] = pr[ee]
                    
       if nonCoincident:
              # Linear combination of columns
              for ii in range(1,len(odexr)-1):
                     rdex = odexr[ii]
                     cdex = odexc[ii]
                     
                     # merge rows and columns
                     GDMS[:,cdex-1] += GDMS[:,cdex]
                     
                     GDMS[:,rdex] += GDMS[:,rdex-1]
                     
              GDMS = np.delete(GDMS, odexc[1:-1], axis=1)
              domain = gdom
              
              deldex = np.copy(odexr[1:-1])
              deldex[1] -= 1
              GDMS = np.delete(GDMS, np.array(odexr[1:-1]), axis=0)
       else:
              domain = edom
                     
       DDMSA = numericalCleanUp(GDMS)
       
       return DDMSA, domain

# Compute Spectral Element 1st derivative matrix (2 sided coupling)
def computeSpectralElementDerivativeMatrix(dom, NE, nonCoincident, endsLaguerre, ORDER):
              
       N = len(dom)
       # Master grid
       gridLG = [True, False, False, True, False]
       endGrid = [True, False, False, False, True]
       gridCH = [False, True, True, False, False]
       
       isLegendre = True
       isChebyshev = False
       isCompactFD10 = False
       isLaguerreEnd1 = endsLaguerre[0]
       isLaguerreEnd2 = endsLaguerre[1]
       
       CSD_interface = False
       CFD_interface = True
       
       idex = []
       cdex1 = []
       cdex2 = []
       sdom = []
       dmats = []
       NEL = []
       
       def getInterfaceDerivativeCFD(ddm1, cdex):
              
              df = 0.5
              di = ddm1[cdex,:]
              di[cdex] *= df
              dic = np.insert(di, cdex+1, di[cdex])
              
              return dic
       
       # Loop over each element in dom (N-1 in 1D)
       for ee in range(N-1):
              # Get the element coordinates
              xa = dom[ee]
              xb = dom[ee+1]
              LE = abs(xb - xa)
              NEL += [NE]
              SDIMS = [xa, xb, LE, NE, NE]
              
              if ee == 0:
                     if isLaguerreEnd1:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom += [-1.0 * np.flip(SREFS[1] + abs(xb))]
                            DMB, CMB, scale = computeLaguerreDerivativeMatrix(SDIMS)
                            RDMS = -1.0 * np.flip(np.flip(DMB,axis=0), axis=1)
                     else:
                            if isLegendre:
                                   SREFS = computeGrid(SDIMS, *gridLG)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                            if isChebyshev:
                                   SREFS = computeGrid(SDIMS, *gridCH)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                            if isCompactFD10:
                                   SREFS = computeGrid(SDIMS, *gridCH)
                                   sdom += [SREFS[1] + xa]
                                   RDMS = computeCompactFiniteDiffDerivativeMatrix1(sdom[ee], ORDER)
                            
              elif ee == N-2:
                     if isLaguerreEnd2:
                            SREFS = computeGrid(SDIMS, *endGrid)
                            sdom += [SREFS[1] + xa]
                            DMB, CMB, scale = computeLaguerreDerivativeMatrix(SDIMS)
                            RDMS = 1.0 * DMB
                     else:
                            if isLegendre:
                                   SREFS = computeGrid(SDIMS, *gridLG)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                            if isChebyshev:
                                   SREFS = computeGrid(SDIMS, *gridCH)
                                   sdom += [SREFS[1] + xa]
                                   RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                            if isCompactFD10:
                                   SREFS = computeGrid(SDIMS, *gridCH)
                                   sdom += [SREFS[1] + xa]
                                   RDMS = computeCompactFiniteDiffDerivativeMatrix1(sdom[ee], ORDER)
              else:
                     # Compute the local derivative matrix
                     if isLegendre:
                            SREFS = computeGrid(SDIMS, *gridLG)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeLegendreDerivativeMatrix(SDIMS)
                     if isChebyshev:
                            SREFS = computeGrid(SDIMS, *gridCH)
                            sdom += [SREFS[1] + xa]
                            RDMS, LTR = computeChebyshevDerivativeMatrix(SDIMS)
                     if isCompactFD10:
                            SREFS = computeGrid(SDIMS, *gridCH)
                            sdom += [SREFS[1] + xa]
                            RDMS = computeCompactFiniteDiffDerivativeMatrix1(sdom[ee], ORDER)
              
              # Make a list of element matrices
              dmats += [RDMS] # Raw derivative matrix
              
              if ee == 0:
                     edom = sdom[ee]
                     gdom = sdom[ee]
              else:
                     edom = np.append(edom, sdom[ee][0:])
                     gdom = np.append(gdom, sdom[ee][1:])
              
              # Store indices of interface dofs
              if ee >= 0 and ee < N-2:
                     idex += [len(gdom) - 1]
                     cdex1 += [len(edom) - 1]
                     cdex2 += [len(edom)]
                     
       # Assemble the raw matrix
       odexr = [0]
       odexc = [0]
       GDMS = np.copy(dmats[0])
       for ee in range(1,N-1):
              # Append the next diagonal block
              GDMS = scl.block_diag(GDMS, dmats[ee])
              
              # Get the indices of coincident interfaces (left)
              tdexr = -(NEL[ee]+1)
              tdexc = -(NEL[ee]+1)
              odexr += [GDMS.shape[0] + tdexr]
              odexc += [GDMS.shape[1] + tdexc]
              
       odexr += [GDMS.shape[0] - 1]
       odexc += [GDMS.shape[1] - 1]
       
       # Compute the continuous common derivatives
       if CSD_interface:
              DDMB = computeCompactFiniteDiffDerivativeMatrix1(gdom, 6)
              DDMC, temp = computeCubicSplineDerivativeMatrix(gdom, True, False, False, False, DDMB)
              
       elif CFD_interface:
              DDMC = computeCompactFiniteDiffDerivativeMatrix1(gdom, ORDER)
       
       DDMC = np.insert(DDMC, idex, 0.0, axis=1)
       
       # Couple derivatives at the interfaces
       GDMS[cdex1,:] = DDMC[idex,:]
       GDMS[cdex2,:] = DDMC[idex,:]
       
       #'''
       if nonCoincident:
              # Linear combination of columns
              for ii in range(1,len(odexr)-1):
                     cdex = odexc[ii]
                     
                     # merge rows and columns
                     GDMS[:,cdex-1] += GDMS[:,cdex]
                     
              GDMS = np.delete(GDMS, odexc[1:-1], axis=1)
              domain = gdom
              
              #print(odexr)
              odexr[0] -= 1
              GDMS = np.delete(GDMS, odexr[1:-1], axis=0)
       else:
              domain = edom
                     
       #DDMSA = numericalCleanUp(GDMS)
       
       return GDMS, domain

# Compute Quintic Spline 1st derivative matrix
def computeQuinticSplineDerivativeMatrix(dom, isClamped, isEssential, DDM_BC):
       
       DM2 = DDM_BC.dot(DDM_BC)
       DM3 = DDM_BC.dot(DM2)
       DM4 = DDM_BC.dot(DM3)
       
       # Initialize matrix blocks
       N = len(dom)
       A = np.zeros((N,N)) # coefficients to 4th derivatives
       B = np.zeros((N,N)) # coefficients to RHS of 4th derivatives
       C = np.zeros((N,N)) # coefficients to 1st derivatives
       D = np.zeros((N,N)) # coefficients to additive part of 1st derivatives
       
       def computeIntegratalConstantMatrices(ii, x):
              
              #x = np.array(x, dtype=np.longdouble)
              
              hp = abs(x[ii+1] - x[ii])
              hp1 = abs(x[ii+2] - x[ii+1])
              hm = abs(x[ii] - x[ii-1])
              
              V = np.zeros((9,9))
              
              a0 = 0.5
              a1 = 1.0 / 6.0
              a2 = 1.0 / 24.0
              a3 = 1.0 / 120.0
              
              xim = x[ii-1]
              xi = x[ii]
              xi2 = a0*xi**2
              xip = x[ii+1]
              xip2 = a0*xip**2
              xir = x[ii+2]
              
              xqm = a1 * (xim**2 + xi * xim + xi**2)
              xqp = a1 * (xi**2 + xip * xi + xip**2)
              xqr = a1 * (xip**2 + xir * xip + xir**2)
              
              # A_j-1, B_j-1, C_j-1, A_j, B_j, C_j, A_j+1, B_j+1, C_j+1 
              V[0,:] = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
              V[1,:] = np.array([-xi, -1.0, 0.0, xi, 1.0, 0.0, 0.0, 0.0, 0.0])
              V[2,:] = np.array([-xi2, -xi, -1.0, xi2, xi, 1.0, 0.0, 0.0, 0.0])
              V[3,:] = np.array([xqm, a0 * (xi + xim), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
              V[4,:] = np.array([0.0, 0.0, 0.0, xqp, a0 * (xip + xi), 1.0, 0.0, 0.0, 0.0])
              V[5,:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, xqr, a0 * (xir + xip), 1.0])
              V[6,:] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
              V[7,:] = np.array([0.0, 0.0, 0.0, xip, 1.0, 0.0, -xip, -1.0, 0.0])
              V[8,:] = np.array([0.0, 0.0, 0.0, xip2, xip, 1.0, -xip2, -xip, -1.0])
              
              rho = np.zeros((9,4))
              rho[0,1] = a0 * (hp + hm)
              rho[1,1] = a1 * (hm**2 - hp**2)
              rho[2,1] = a2 * (hm**3 + hp**3)
              rho[3,0] = a3 * hm**3
              rho[3,1] = -a3 * hm**3
              rho[4,1] = a3 * hp**3
              rho[4,2] = -a3 * hp**3
              rho[5,2] = a3 * hp1**3
              rho[5,3] = -a3 * hp1**3
              rho[6,2] = -a0 * (hp + hp1)
              rho[7,2] = a1 * (hp1**2 - hp**2)
              rho[8,2] = -a2 * (hp**3 + hp1**3)
              
              eta = np.zeros((9,4))
              eta[3,:] = 1.0 / hm * np.array([-1.0, 1.0, 0.0, 0.0])
              eta[4,:] = 1.0 / hp * np.array([0.0, -1.0, 1.0, 0.0])
              eta[5,:] = 1.0 / hp1 *np.array([0.0, 0.0, -1.0, 1.0])
              
              Q, R = scl.qr(V)
              PLU = scl.lu_factor(R)
              VI = scl.lu_solve(PLU, Q.T)
              
              OM = VI.dot(rho)
              ET = VI.dot(eta)
                            
              return OM, ET
                     
       # Loop over each interior point in the irregular grid
       for ii in range(1,N-1):
              hp = abs(dom[ii+1] - dom[ii])
              hm = abs(dom[ii] - dom[ii-1])
              
              #%% LHS matrix
              a0 = 0.5
              a1 = 1.0 / 6.0
              a2 = 1.0 / 24.0
              hc = hp + hm
              
              #'''
              if ii == 1:                     
                     # Compute the right EIC
                     OM, ET = computeIntegratalConstantMatrices(ii, dom)
                     
                     # Assemble to the equation for Z
                     A[ii,ii] = a0 * hc
                     A[ii,ii-1:ii+3] += OM[3,:]
                     #A[ii,ii-1:ii+2] -= OM[0,1:]
                     
                     B[ii,ii-1:ii+3] += ET[3,:]
                     #B[ii,ii-1:ii+2] -= ET[0,1:]
                     
                     # Compute the C matrix (coefficients to Z)
                     # Compute the D matrix (coefficients to Q)
                     
                     C[ii,ii] = -a2 * hp**3
                     C[ii,ii-1:ii+3] += a0 * dom[ii]**2 * OM[3,:] + dom[ii] * OM[4,:] + OM[5,:]
                     D[ii,ii-1:ii+3] += a0 * dom[ii]**2 * ET[3,:] + dom[ii] * ET[4,:] + ET[5,:]
                     
                     C[ii-1,ii-1] = -a2 * hm**3
                     C[ii-1,ii-1:ii+3] += a0 * dom[ii-1]**2 * OM[0,:] + dom[ii-1] * OM[1,:] + OM[2,:]
                     D[ii-1,ii-1:ii+3] += a0 * dom[ii-1]**2 * ET[0,:] + dom[ii-1] * ET[1,:] + ET[2,:]
                     
              elif ii == N-2:
                     # Compute the left EIC
                     OM, ET = computeIntegratalConstantMatrices(ii-1, dom)
                     
                     # Assemble to the equation for Z
                     A[ii,ii] = a0 * hc
                     #A[ii,ii-1:ii+2] += OM[6,:-1]
                     A[ii,ii-2:ii+2] -= OM[3,:]
                     
                     #B[ii,ii-1:ii+2] += ET[6,:-1]
                     B[ii,ii-2:ii+2] -= ET[3,:]
                     
                     # Compute the C matrix (coefficients to Z)
                     # Compute the D matrix (coefficients to Q)
                     
                     C[ii,ii] = +a2 * hm**3
                     C[ii,ii-2:ii+2] += a0 * dom[ii]**2 * OM[3,:] + dom[ii] * OM[4,:] + OM[5,:]
                     D[ii,ii-2:ii+2] += a0 * dom[ii]**2 * ET[3,:] + dom[ii] * ET[4,:] + ET[5,:]
                     
                     C[ii+1,ii+1] = +a2 * hp**3
                     C[ii+1,ii-2:ii+2] += a0 * dom[ii+1]**2 * OM[6,:] + dom[ii+1] * OM[7,:] + OM[8,:]
                     D[ii+1,ii-2:ii+2] += a0 * dom[ii+1]**2 * ET[6,:] + dom[ii+1] * ET[7,:] + ET[8,:]
              else:
                     # Compute adjacent EIC and assemble to internal equations for Z
                     OM1, ET1 = computeIntegratalConstantMatrices(ii-1, dom)
                     OM, ET = computeIntegratalConstantMatrices(ii, dom)
                     
                     A[ii,ii] = a0 * hc
                     A[ii,ii-1:ii+3] += OM[3,:]
                     A[ii,ii-2:ii+2] -= OM1[3,:]
                     
                     B[ii, ii-1:ii+3] += ET[3,:]
                     B[ii, ii-2:ii+2] -= ET1[3,:]
                     
                     # Compute the C matrix (coefficients to Z)
                     C[ii,ii] = -a2 * hp**3
                     C[ii,ii-1:ii+3] += a0 * dom[ii]**2 * OM[3,:] + dom[ii] * OM[4,:] + OM[5,:]
                     
                     # Compute the D matrix (coefficients to Q)
                     D[ii,ii-1:ii+3] += a0 * dom[ii]**2 * ET[3,:] + dom[ii] * ET[4,:] + ET[5,:]
              #'''
              
       # Prescribed derivative conditions
       D4A = DM4[0,:] # left end 4th derivative
       D4B = DM4[-1,:] # right end 4th derivative
       D1A = DDM_BC[0,:] # left end 1st derivative
       D1B = DDM_BC[-1,:] # right end 1st derivative
       
       if isClamped:
              A[0,0] = 1.0
              B[0,:] = D4A
              A[-1,-1] = 1.0
              B[-1,:] = D4B
              
              Q, R = scl.qr(A)
              PLU = scl.lu_factor(R)
              AIB = scl.lu_solve(PLU, (Q.T).dot(B))
       elif isEssential:
              A[0,0] = -1.0
              A[0,1] = +1.0
              B[0,:] = np.zeros(N)
              
              A[N-1,N-2] = -1.0
              A[N-1,N-1] = +1.0
              B[N-1,:] = np.zeros(N)
              
              Q, R = scl.qr(A)
              PLU = scl.lu_factor(R)
              AIB = scl.lu_solve(PLU, (Q.T).dot(B))
       else:
              # NATURAL cubic spline.
              AIB = np.diag(np.ones(N))
              AIB[1:N-1,1:N-1] = np.linalg.solve(A[1:-1,1:-1], B[1:-1,1:-1])
       
       # Compute the 1st derivative matrix
       DDM = C.dot(AIB) + D

       # Set boundary derivatives from specified
       #if isClamped:
       #       DDM[0,:] = D1A
       #       DDM[-1,:] = D1B
              
       DDM = numericalCleanUp(DDM)
       #DDM = removeLeastSingularValue(DDM)
       return DDM, AIB
   
# Compute Quartic Spline 1st derivative matrix
def computeQuarticSplineDerivativeMatrix(dom, isClamped, isEssential, DDM_BC):
       
       a0 = 0.5
       a1 = 1.0 / 6.0
       a2 = 1.0 / 24.0
       
       DM2 = DDM_BC.dot(DDM_BC)
       DM3 = DDM_BC.dot(DM2)
       
       # Initialize matrix blocks
       N = len(dom)
       A = np.zeros((N,N)) # coefficients to 2nd derivatives
       B = np.zeros((N,N)) # coefficients to RHS of 2nd derivatives
       C = np.zeros((N,N)) # coefficients to 1st derivatives
       D = np.zeros((N,N)) # coefficients to additive part of 1st derivatives
       
       def computeIntegratalConstantMatrices(ii, x):
                            
              hj = (x[ii+1] - x[ii])
              hm = (x[ii] - x[ii-1])
              hp = (x[ii+2] - x[ii+1])
              
              V = np.zeros((6,6))
              
              # A_j-1, B_j-1, A_j, B_j, A_j+1, B_j+1 
              V[0,:] = np.array([-1.0, 0.0, +1.0, 0.0, 0.0, 0.0])
              V[1,:] = np.array([0.0, 0.0, +1.0, 0.0, -1.0, 0.0])
              V[2,:] = np.array([-hm, -1.0, 0.0, +1.0, 0.0, 0.0])
              V[3,:] = np.array([0.0, 0.0, +hj, +1.0, 0.0, -1.0])
              V[4,:] = np.array([a0 * hm**2, +hm, 0.0, 0.0, 0.0, 0.0])
              V[5,:] = np.array([0.0, 0.0, a0 * hj**2, +hj, 0.0, 0.0])
              
              # Z_i-1, Z_i, Z_i+1
              rho = np.zeros((6,3))
              rho[0,1] = a0 * (hm + hj)
              rho[1,2] = -a0 * (hp + hj)
              rho[2,1] = a1 * (hm - hj) * (hm + hj)
              rho[3,2] = a1 * (hp - hj) * (hp + hj)
              rho[4,0] = -a2 * hm**3
              rho[4,1] = -a2 * hj**3
              rho[5,1] = -a2 * hj**3
              rho[5,2] = -a2 * hp**3
              
              # q_i-1, q_i, q_i+1
              eta = np.zeros((6,3))
              eta[4,:] = np.array([-1.0, +1.0, 0.0])
              eta[5,:] = np.array([0.0, -1.0, +1.0])
              
              Q, R = scl.qr(V)
              PLU = scl.lu_factor(R)
              VI = scl.lu_solve(PLU, Q.T)
              
              OM = VI.dot(rho)
              ET = VI.dot(eta)
                            
              return OM, ET
          
       # Loop over each interior point in the irregular grid
       for ii in range(1,N-1):
              hp = abs(dom[ii+1] - dom[ii])
              hm = abs(dom[ii] - dom[ii-1])
              
              #%% LHS matrix
              a0 = 0.5
              a1 = 1.0 / 6.0
              hc = hp + hm
              
              #'''
              if ii == 1:                     
                     # Compute the right EIC
                     OM, ET = computeIntegratalConstantMatrices(ii, dom)
                     
                     # Assemble to the equation for Z
                     A[ii,ii] = a0 * hc
                     A[ii,ii-1:ii+2] += OM[2,:]
                     #A[ii,ii-1:ii+1] -= OM[0,1:]
                     
                     B[ii,ii-1:ii+2] += ET[2,:]
                     #B[ii,ii-1:ii+2] += +(ET2[0,1:])
                     
                     # Compute the C matrix (coefficients to Z)
                     # Compute the D matrix (coefficients to Q)
                     
                     C[ii,ii] = a1 * hp**2
                     C[ii,ii-1:ii+2] += OM[3,:]
                     D[ii,ii-1:ii+2] += ET[3,:]
                     
                     C[ii-1,ii-1] = a1 * hm**2
                     C[ii-1,ii-1:ii+2] += OM[1,:]
                     D[ii-1,ii-1:ii+2] += ET[1,:]
                     
              elif ii == N-2:
                     # Compute the left EIC
                     OM, ET = computeIntegratalConstantMatrices(ii-1, dom)
                     
                     # Assemble to the equation for Z
                     A[ii,ii] = a0 * hc
                     #A[ii,ii:ii+2] += OM[4,:-1]
                     A[ii,ii-1:ii+2] += OM[4,:]
                     
                     #B[ii,ii-1:ii+2] += -ET1[6,:-1]
                     B[ii,ii-1:ii+2] += ET[4,:]
                     
                     # Compute the C matrix (coefficients to Z)
                     # Compute the D matrix (coefficients to Q)
                     
                     C[ii,ii] = a1 * hm**2
                     C[ii,ii-1:ii+2] += hm * OM[2,:] + OM[3,:]
                     D[ii,ii-1:ii+2] += hm * ET[2,:] + ET[3,:]
                     
                     C[ii+1,ii+1] = a1 * hp**2
                     C[ii+1,ii-1:ii+2] += hp * OM[4,:] + OM[5,:]
                     D[ii+1,ii-1:ii+2] += hp * ET[4,:] + ET[5,:]
              else:
                     # Compute adjacent EIC and assemble to internal equations for Z
                     OM1, ET1 = computeIntegratalConstantMatrices(ii-1, dom)
                     OM, ET = computeIntegratalConstantMatrices(ii, dom)
                     
                     A[ii,ii] = a0 * hc
                     A[ii, ii-1:ii+2] += OM[2,:]
                     A[ii, ii-2:ii+1] -= OM1[2,:]
                     
                     B[ii, ii-1:ii+2] += ET[2,:]
                     B[ii, ii-2:ii+1] -= ET1[2,:]
                     
                     # Compute the C matrix (coefficients to Z)
                     C[ii,ii] = a1 * hp**2
                     C[ii,ii-1:ii+2] += OM[3,:]
                     
                     # Compute the D matrix (coefficients to Q)
                     D[ii,ii-1:ii+2] += ET[3,:]
              
       # Prescribed derivative conditions
       D3A = DM3[0,:] # left end 3th derivative
       D3B = DM3[-1,:] # right end 3th derivative
       D1A = DDM_BC[0,:] # left end 1st derivative
       D1B = DDM_BC[-1,:] # right end 1st derivative
              
       if isClamped:
              A[0,0] = 1.0
              B[0,:] = D3A
              A[-1,-1] = 1.0
              B[-1,:] = D3B
              
              Q, R = scl.qr(A)
              PLU = scl.lu_factor(R)
              AIB = scl.lu_solve(PLU, (Q.T).dot(B))
       elif isEssential:
              A[0,0] = -1.0
              A[0,1] = +1.0
              B[0,:] = 0.0
              
              A[N-1,N-2] = -1.0
              A[N-1,N-1] = +1.0
              B[N-1,:] = 0.0
                            
              Q, R = scl.qr(A)
              PLU = scl.lu_factor(R)
              AIB = scl.lu_solve(PLU, (Q.T).dot(B))
       else:
              # NATURAL cubic spline.
              AIB = np.diag(np.ones(N))
              AIB[1:N-1,1:N-1] = np.linalg.solve(A[1:-1,1:-1], B[1:-1,1:-1])
       
       # Compute the 1st derivative matrix
       DDM = C.dot(AIB) + D
              
       DDM = numericalCleanUp(DDM)
       #DDM = removeLeastSingularValue(DDM)
       
       return DDM, AIB

# Computes Cubic Spline 1st derivative matrix
def computeCubicSplineDerivativeMatrix(dom, isClamped, isEssential, DDM_BC):
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
              
       for ii in range(0,N-1):
              hp = abs(dom[ii+1] - dom[ii])
              C[ii,ii] = -1.0 / 3.0 * hp
              C[ii,ii+1] = -1.0 / 6.0 * hp
              
              D[ii,ii] = -1.0 / hp
              D[ii,ii+1] = +1.0 / hp
              
       hn = abs(dom[-1] - dom[-2])
       
       C[N-1,N-1] = -1.0 / 3.0 * hn
       C[N-1,N-2] = -1.0 / 6.0 * hn
       
       D[N-1,N-1] = +1.0 / hn
       D[N-1,N-2] = -1.0 / hn
       
       if isClamped:
              
              if np.isscalar(DDM_BC):
                     DM2 = 0.0
                     B[0,:] = DM2
                     B[-1,:] = DM2
              else:
                     DM2 = DDM_BC.dot(DDM_BC)
                     B[0,:] = DM2[0,:]
                     B[-1,:] = DM2[-1,:]
              
              A[0,0] = 1.0
              A[-1,-1] = 1.0
              
              Q, R = scl.qr(A)
              PLU = scl.lu_factor(R)
              AIB = scl.lu_solve(PLU, (Q.T).dot(B))
       elif isEssential:
              A[0,0] = +1.0
              A[0,1] = -1.0
              B[0,:] = np.zeros(N)
              
              A[N-1,N-2] = -1.0
              A[N-1,N-1] = +1.0
              B[N-1,:] = np.zeros(N)
              
              Q, R = scl.qr(A)
              PLU = scl.lu_factor(R)
              AIB = scl.lu_solve(PLU, (Q.T).dot(B))
       else:
              # NATURAL cubic spline.
              AIB = np.diag(np.ones(N))
              AIB[1:N-1,1:N-1] = np.linalg.solve(A[1:-1,1:-1], B[1:-1,1:-1])
       
       # Compute the 1st derivative matrix
       DDM = C.dot(AIB) + D
              
       DDM = numericalCleanUp(DDM)
       #DDM = removeLeastSingularValue(DDM)
       return DDM, AIB

# Computes standard 4th order compact finite difference 1st derivative matrix
def computeCompactFiniteDiffDerivativeMatrix1(dom, order):
       
       end3 = False
       if order == 4:
              end4 = True
              end6 = False
              end8 = False
       elif order == 6:
              end4 = False
              end6 = True
              end8 = False
       elif order > 6:
              end4 = False
              end6 = False
              end8 = True
       
       # Initialize the left and right derivative matrices
       N = len(dom)
       LDM = np.zeros((N,N)) # tridiagonal
       RDM = np.zeros((N,N)) # centered difference
       
       def p2Matrix6(hm1, hp1, hp2, hp3, hp4):
              CM = np.ones((6,6))
              c1 = hp1; c2 = hm1
              c3 = hp2 + hp1
              c5 = hp3 + hp2 + hp1
              c7 = hp4 + hp3 + hp2 + hp1

              # One sided boundary scheme (2 forward derivatives)
              CM[0,:] = [+c1, -c2, +c3, +c5, +c7, -2.0]
              CM[1,:] = [c1**2, +c2**2, +c3**2, +c5**2, +c7**2, -2.0 * (hp1 - hm1)]
              CM[2,:] = [c1**3, -c2**3, +c3**3, +c5**3, +c7**3, -3.0 * (hp1**2 + hm1**2)]
              CM[3,:] = [c1**4, +c2**4, +c3**4, +c5**4, +c7**4, -4.0 * (hp1**3 - hm1**3)]
              CM[4,:] = [c1**5, -c2**5, +c3**5, +c5**5, +c7**5, -5.0 * (hp1**4 + hm1**4)]
              CM[5,:] = [c1**6, +c2**6, +c3**6, +c5**6, +c7**6, -6.0 * (hp1**5 - hm1**5)]

              return CM
       
       def p2Matrix8(hm1, hp1, hp2, hp3, hp4, hp5, hp6):
              CM = np.ones((8,8))
              c1 = hp1; c2 = hm1
              c3 = hp2 + hp1
              c5 = hp3 + hp2 + hp1
              c7 = hp4 + hp3 + hp2 + hp1
              c9 = hp5 + hp4 + hp3 + hp2 + hp1
              c11 = hp6 + hp5 + hp4 + hp3 + hp2 + hp1

              # One sided boundary scheme (2 forward derivatives)
              CM[0,:] = [+c1, -c2, +c3, +c5, +c7, +c9, +c11, -2.0]
              CM[1,:] = [c1**2, +c2**2, +c3**2, +c5**2, +c7**2, +c9**2, +c11**2, -2.0 * (hp1 - hm1)]
              CM[2,:] = [c1**3, -c2**3, +c3**3, +c5**3, +c7**3, +c9**3, +c11**3, -3.0 * (hp1**2 + hm1**2)]
              CM[3,:] = [c1**4, +c2**4, +c3**4, +c5**4, +c7**4, +c9**4, +c11**4, -4.0 * (hp1**3 - hm1**3)]
              CM[4,:] = [c1**5, -c2**5, +c3**5, +c5**5, +c7**5, +c9**5, +c11**5, -5.0 * (hp1**4 + hm1**4)]
              CM[5,:] = [c1**6, +c2**6, +c3**6, +c5**6, +c7**6, +c9**6, +c11**6, -6.0 * (hp1**5 - hm1**5)]
              CM[6,:] = [c1**7, -c2**7, +c3**7, +c5**7, +c7**7, +c9**7, +c11**7, -7.0 * (hp1**6 + hm1**6)]
              CM[7,:] = [c1**8, +c2**8, +c3**8, +c5**8, +c7**8, +c9**8, +c11**8, -8.0 * (hp1**7 - hm1**7)]

              return CM
       
       def endMatrix4(hp1, hp2, hp3):
              CM = np.ones((4,4))
              c1 = hp1
              c3 = hp2 + hp1
              c5 = hp3 + hp2 + hp1

              # One sided boundary scheme (2 forward derivatives)
              CM[0,:] = [+c1, +c3, +c5, -1.0]
              CM[1,:] = [c1**2, +c3**2, +c5**2, -2.0 * c1]
              CM[2,:] = [c1**3, +c3**3, +c5**3, -3.0 * c1**2]
              CM[3,:] = [c1**4, +c3**4, +c5**4, -4.0 * c1**3]
              
              return CM
       
       def endMatrix6(hp1, hp2, hp3, hp4, hp5):
              CM = np.ones((6,6))
              c1 = hp1
              c3 = hp2 + hp1
              c5 = hp3 + hp2 + hp1
              c7 = hp4 + hp3 + hp2 + hp1
              c9 = hp5 + hp4 + hp3 + hp2 + hp1

              # One sided boundary scheme (2 forward derivatives)
              CM[0,:] = [+c1, +c3, +c5, +c7, +c9, -1.0]
              CM[1,:] = [c1**2, +c3**2, +c5**2, +c7**2, +c9**2, -2.0 * c1]
              CM[2,:] = [c1**3, +c3**3, +c5**3, +c7**3, +c9**3, -3.0 * c1**2]
              CM[3,:] = [c1**4, +c3**4, +c5**4, +c7**4, +c9**4, -4.0 * c1**3]
              CM[4,:] = [c1**5, +c3**5, +c5**5, +c7**5, +c9**5, -5.0 * c1**4]
              CM[5,:] = [c1**6, +c3**6, +c5**6, +c7**6, +c9**6, -6.0 * c1**5]

              return CM
       
       def endMatrix8(hp1, hp2, hp3, hp4, hp5, hp6, hp7):
              CM = np.ones((8,8))
              c1 = hp1
              c3 = hp2 + hp1
              c5 = hp3 + hp2 + hp1
              c7 = hp4 + hp3 + hp2 + hp1
              c9 = hp5 + hp4 + hp3 + hp2 + hp1
              c11 = hp6 + hp5 + hp4 + hp3 + hp2 + hp1
              c13 = hp7 + hp6 + hp5 + hp4 + hp3 + hp2 + hp1

              # One sided boundary scheme (2 forward derivatives)
              CM[0,:] = [+c1, +c3, +c5, +c7, +c9, +c11, +c13, -1.0]
              CM[1,:] = [c1**2, +c3**2, +c5**2, +c7**2, +c9**2, +c11**2, +c13**2, -2.0 * c1]
              CM[2,:] = [c1**3, +c3**3, +c5**3, +c7**3, +c9**3, +c11**3, +c13**3, -3.0 * c1**2]
              CM[3,:] = [c1**4, +c3**4, +c5**4, +c7**4, +c9**4, +c11**4, +c13**4, -4.0 * c1**3]
              CM[4,:] = [c1**5, +c3**5, +c5**5, +c7**5, +c9**5, +c11**5, +c13**5, -5.0 * c1**4]
              CM[5,:] = [c1**6, +c3**6, +c5**6, +c7**6, +c9**6, +c11**6, +c13**6, -6.0 * c1**5]
              CM[6,:] = [c1**7, +c3**7, +c5**7, +c7**7, +c9**7, +c11**7, +c13**7, -7.0 * c1**6]
              CM[7,:] = [c1**8, +c3**8, +c5**8, +c7**8, +c9**8, +c11**8, +c13**8, -8.0 * c1**7]

              return CM
       
       def interiorMatrix10(hm3, hm2, hm1, hp1, hp2, hp3, ND):
              CM = np.ones((ND,ND))
              c1 = hp1; c2 = hm1
              c3 = hp2 + hp1; c4 = hm1 + hm2
              c5 = hp3 + hp2 + hp1; c6 = hm1 + hm2 + hm3

              #''' Pentadiagonal left, Septadiagonal right
              CM[0,:] = [c1, -c2, +c3, -c4, c5, -c6, -1.0, -1.0, -1.0, -1.0]
              CM[1,:] = [c1**2, +c2**2, +c3**2, +c4**2, +c5**2, +c6**2, +2.0 * c2, -2.0 * c1, +2.0 * c4, -2.0 * c3]
              CM[2,:] = [c1**3, -c2**3, +c3**3, -c4**3, +c5**3, -c6**3, -3.0 * c2**2, -3.0 * c1**2, -3.0 * c4**2, -3.0 * c3**2]
              CM[3,:] = [c1**4, +c2**4, +c3**4, +c4**4, +c5**4, +c6**4, +4.0 * c2**3, -4.0 * c1**3, +4.0 * c4**3, -4.0 * c3**3]
              CM[4,:] = [c1**5, -c2**5, +c3**5, -c4**5, +c5**5, -c6**5, -5.0 * c2**4, -5.0 * c1**4, -5.0 * c4**4, -5.0 * c3**4]
              CM[5,:] = [c1**6, +c2**6, +c3**6, +c4**6, +c5**6, +c6**6, +6.0 * c2**5, -6.0 * c1**5, +6.0 * c4**5, -6.0 * c3**5]
              CM[6,:] = [c1**7, -c2**7, +c3**7, -c4**7, +c5**7, -c6**7, -7.0 * c2**6, -7.0 * c1**6, -7.0 * c4**6, -7.0 * c3**6]
              CM[7,:] = [c1**8, +c2**8, +c3**8, +c4**8, +c5**8, +c6**8, +8.0 * c2**7, -8.0 * c1**7, +8.0 * c4**7, -8.0 * c3**7]
              CM[8,:] = [c1**9, -c2**9, +c3**9, -c4**9, +c5**9, -c6**9, -9.0 * c2**8, -9.0 * c1**8, -9.0 * c4**8, -9.0 * c3**8]
              CM[9,:] = [c1**10, +c2**10, +c3**10, +c4**10, +c5**10, +c6**10, +10.0 * c2**9, -10.0 * c1**9, +10.0 * c4**9, -10.0 * c3**9]
                            
              return CM
              
       for ii in range(1,N-1):
              # Get the metric weights
              hp1 = abs(dom[ii+1] - dom[ii])
              hm1 = abs(dom[ii] - dom[ii-1])
              
              if ii in range(3,N-3):
                     hp2 = abs(dom[ii+2] - dom[ii+1])
                     hm2 = abs(dom[ii-1] - dom[ii-2])
                     hp3 = abs(dom[ii+3] - dom[ii+2])
                     hm3 = abs(dom[ii-3] - dom[ii-2])
              elif ii in range(2,N-2):
                     hp2 = abs(dom[ii+2] - dom[ii+1])
                     hm2 = abs(dom[ii-1] - dom[ii-2])
                     hp3 = 0.0
                     hm3 = 0.0
              else:
                     hp2 = 0.0
                     hm2 = 0.0
                     hp3 = 0.0
                     hm3 = 0.0
              
              ND = 10
              CM10 = interiorMatrix10(hm3, hm2, hm1, hp1, hp2, hp3, ND)
              CMV = np.zeros(ND)
              CMV[0] = 1.0
       
              if (order == 4 and ii in range(1,N-1)):
                     
                     alpha = 0.25
                     beta = 0.25
                     
                     # Delete columns
                     ddex = [2, 3, 4, 5, 8, 9]
                     CM4 = np.delete(CM10, ddex, axis=1)
                                     
                     # Delete rows to 4th order
                     ddex = [4, 5, 6, 7, 8, 9]
                     CM4 = np.delete(CM4, ddex, axis=0)
                     CM4_V = np.delete(CMV, ddex, axis=0)
                     
                     # Constraint alpha = beta = 0.25
                     CM4[:,-2] += CM4[:,-1]
                     CM4_V -= alpha * CM4[:,-2]
                     CMS = CM4[0:-2,0:-2]
                     
                     Q, R = scl.qr(CMS)
                     PLU = scl.lu_factor(R)
                     CF = scl.lu_solve(PLU, (Q.T).dot(CM4_V[0:-2]))
                     CFE = -np.sum(CF[0:2])
                     
                     # Write the right equation
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     
                     # Write the left equation
                     LDM[ii,ii-1] = alpha
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = beta
                     
              if order == 6:
                     if ii in [1,N-2]:
                            hm1 = dom[1] - dom[0]
                            hp1 = dom[2] - dom[1]
                            hp2 = dom[3] - dom[2]
                            hp3 = dom[4] - dom[3]
                            hp4 = dom[5] - dom[4]
                            #hp5 = dom[6] - dom[5]
                            #hp6 = dom[7] - dom[6]
                            
                            CME = p2Matrix6(hm1, hp1, hp2, hp3, hp4)#, hp5, hp6)
                                            
                            CME_V = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])#, 0.0, 0.0])
                            
                            Q, R = scl.qr(CME)
                            PLU = scl.lu_factor(R)
                            CF = scl.lu_solve(PLU, (Q.T).dot(CME_V))
                            CFE = -np.sum(CF[0:-1])
                            
                            # Write the right equation
                            if ii == 1 or ii == 2:
                                   RDM[ii,ii-1] = CF[1]
                                   RDM[ii,ii] = CFE
                                   RDM[ii,ii+1] = CF[0]
                                   RDM[ii,ii+2] = CF[2]
                                   RDM[ii,ii+3] = CF[3]
                                   RDM[ii,ii+4] = CF[4]
                                   #RDM[ii,ii+5] = CF[5]
                                   #RDM[ii,ii+6] = CF[6]
                            elif ii == N-3 or ii == N-2:
                                   RDM[ii,ii+1] = -CF[1]
                                   RDM[ii,ii] = -CFE
                                   RDM[ii,ii-1] = -CF[0]
                                   RDM[ii,ii-2] = -CF[2]
                                   RDM[ii,ii-3] = -CF[3]
                                   RDM[ii,ii-4] = -CF[4]
                                   #RDM[ii,ii-5] = -CF[5]
                                   #RDM[ii,ii-6] = -CF[6]
                            
                            # Write the left equation
                            LDM[ii,ii-1] = CF[-1]
                            LDM[ii,ii] = 1.0
                            LDM[ii,ii+1] = CF[-1]
                            
                     elif ii in range(2,N-2):   
                                
                             alpha = 1.0 / 3.0 
                             beta = 1.0 / 3.0
                             
                             # Delete columns
                             ddex = [4, 5, 8, 9]
                             CM6 = np.delete(CM10, ddex, axis=1) 
                             
                             # Delete rows to 6th order
                             ddex = [6, 7, 8, 9]
                             CM6 = np.delete(CM6, ddex, axis=0)
                             CM6_V = np.delete(CMV, ddex, axis=0)
                             
                             # Constraint alpha = beta = 1/3
                             CM6[:,-2] += CM6[:,-1]
                             CM6_V -= alpha * CM6[:,-2]
                             CMS = CM6[0:-2,0:-2]
                             
                             Q, R = scl.qr(CMS)
                             PLU = scl.lu_factor(R)
                             CF = scl.lu_solve(PLU, (Q.T).dot(CM6_V[0:-2]))
                             CFE = -np.sum(CF[0:4])
                             
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
                             
              if order == 8 or order == 10:
                     if ii in [1,N-2]:
                            hm1 = dom[1] - dom[0]
                            hp1 = dom[2] - dom[1]
                            hp2 = dom[3] - dom[2]
                            hp3 = dom[4] - dom[3]
                            hp4 = dom[5] - dom[4]
                            hp5 = dom[6] - dom[5]
                            hp6 = dom[7] - dom[6]
                            
                            CME = p2Matrix8(hm1, hp1, hp2, hp3, hp4, hp5, hp6)
                                            
                            CME_V = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                            
                            Q, R = scl.qr(CME)
                            PLU = scl.lu_factor(R)
                            CF = scl.lu_solve(PLU, (Q.T).dot(CME_V))
                            CFE = -np.sum(CF[0:-1])
                            
                            # Write the right equation
                            if ii == 1 or ii == 2:
                                   RDM[ii,ii-1] = CF[1]
                                   RDM[ii,ii] = CFE
                                   RDM[ii,ii+1] = CF[0]
                                   RDM[ii,ii+2] = CF[2]
                                   RDM[ii,ii+3] = CF[3]
                                   RDM[ii,ii+4] = CF[4]
                                   RDM[ii,ii+5] = CF[5]
                                   RDM[ii,ii+6] = CF[6]
                            elif ii == N-3 or ii == N-2:
                                   RDM[ii,ii+1] = -CF[1]
                                   RDM[ii,ii] = -CFE
                                   RDM[ii,ii-1] = -CF[0]
                                   RDM[ii,ii-2] = -CF[2]
                                   RDM[ii,ii-3] = -CF[3]
                                   RDM[ii,ii-4] = -CF[4]
                                   RDM[ii,ii-5] = -CF[5]
                                   RDM[ii,ii-6] = -CF[6]
                            
                            # Write the left equation
                            LDM[ii,ii-1] = CF[-1]
                            LDM[ii,ii] = 1.0
                            LDM[ii,ii+1] = CF[-1]
                     if ii in [2,N-3]:
                            CMI = np.copy(CM10)
                            CMI[:,-2] += CMI[:,-1]
                            CMI[:,-4] += CMI[:,-3]
                            CM10_V = CMV #- (alpha * CMI[:,-4] + theta * CMI[:,-2])
                            
                            sdex = np.array([0, 1, 2, 3, -2, -4])
                            CMS = CMI[np.ix_(sdex,sdex)]
                            
                            Q, R = scl.qr(CMS)
                            PLU = scl.lu_factor(R)
                            CF = scl.lu_solve(PLU, (Q.T).dot(CM10_V[sdex]))
                            CFE = -np.sum(CF[0:4])
                            
                            # Write the right equation
                            RDM[ii,ii-2] = CF[3]
                            RDM[ii,ii-1] = CF[1]
                            RDM[ii,ii] = CFE
                            RDM[ii,ii+1] = CF[0]
                            RDM[ii,ii+2] = CF[2]
                            
                            # Write the left equation
                            LDM[ii,ii-2] = CF[4]
                            LDM[ii,ii-1] = CF[5]
                            LDM[ii,ii] = 1.0
                            LDM[ii,ii+1] = CF[5]
                            LDM[ii,ii+2] = CF[4]
                            
              if (order == 8 and ii in range(3,N-3)):
                                
                             alpha = 2.0 / 5.0 
                             beta = 2.0 / 5.0
                             
                             # Delete columns
                             ddex = [8, 9]
                             CM8 = np.delete(CM10, ddex, axis=1) 
                             
                             # Delete rows to 8th order
                             ddex = [8, 9]
                             CM8 = np.delete(CM8, ddex, axis=0)
                             CM8_V = np.delete(CMV, ddex, axis=0)
                             
                             # Constraint alpha = beta = 2/5
                             CM8[:,-2] += CM8[:,-1]
                             CM8_V -= alpha * CM8[:,-2]
                             CMS = CM8[0:-2,0:-2]
                             
                             Q, R = scl.qr(CMS)
                             PLU = scl.lu_factor(R)
                             CF = scl.lu_solve(PLU, (Q.T).dot(CM8_V[0:-2]))
                             CFE = -np.sum(CF[0:6])
                             
                             # Write the right equation
                             RDM[ii,ii-3] = CF[5]
                             RDM[ii,ii-2] = CF[3]
                             RDM[ii,ii-1] = CF[1]
                             RDM[ii,ii] = CFE
                             RDM[ii,ii+1] = CF[0]
                             RDM[ii,ii+2] = CF[2]
                             RDM[ii,ii+3] = CF[4]
                             
                             # Write the left equation
                             LDM[ii,ii-1] = alpha
                             LDM[ii,ii] = 1.0
                             LDM[ii,ii+1] = beta
                     
              # Loop over each interior point in the irregular grid
              if (order == 10 and ii in range(3,N-3)):
                     
                     CMI = np.copy(CM10)
                     CMI[:,-2] += CMI[:,-1]
                     CMI[:,-4] += CMI[:,-3]
                     CM10_V = CMV #- (alpha * CMI[:,-4] + theta * CMI[:,-2])
                     
                     sdex = np.array([0, 1, 2, 3, 4, 5, -2, -4])
                     CMS = CMI[np.ix_(sdex,sdex)]
                     
                     Q, R = scl.qr(CMS)
                     PLU = scl.lu_factor(R)
                     CF = scl.lu_solve(PLU, (Q.T).dot(CM10_V[sdex]))
                     CFE = -np.sum(CF[0:6])
                     
                     # Write the right equation
                     RDM[ii,ii-3] = CF[5]
                     RDM[ii,ii-2] = CF[3]
                     RDM[ii,ii-1] = CF[1]
                     RDM[ii,ii] = CFE
                     RDM[ii,ii+1] = CF[0]
                     RDM[ii,ii+2] = CF[2]
                     RDM[ii,ii+3] = CF[4]
                     
                     # Write the left equation
                     LDM[ii,ii-2] = CF[6]
                     LDM[ii,ii-1] = CF[7]
                     LDM[ii,ii] = 1.0
                     LDM[ii,ii+1] = CF[7]
                     LDM[ii,ii+2] = CF[6]
                     
       if end8: 
              # Coefficients for 6th order compact one-sided schemes
              hp1 = dom[1] - dom[0]
              hp2 = dom[2] - dom[1]
              hp3 = dom[3] - dom[2]
              hp4 = dom[4] - dom[3]
              hp5 = dom[5] - dom[4]
              hp6 = dom[6] - dom[5]
              hp7 = dom[7] - dom[6]
       
              # Compute the stencil coefficients
              CME = endMatrix8(hp1, hp2, hp3, hp4, hp5, hp6, hp7)
              CME_V = np.zeros(8)
              CME_V[0] = 1.0
              
              Q, R = scl.qr(CME)
              PLU = scl.lu_factor(R)
              CF_F = scl.lu_solve(PLU, (Q.T).dot(CME_V))
              beta = CF_F[-1]
              
              LDM[0,0] = 1.0
              LDM[0,1] = beta
              RDM[0,0] = -np.sum(CF_F[0:-1])
              RDM[0,1] = CF_F[0]
              RDM[0,2] = CF_F[1]
              RDM[0,3] = CF_F[2]
              RDM[0,4] = CF_F[3]
              RDM[0,5] = CF_F[4]
              RDM[0,6] = CF_F[5]
              RDM[0,7] = CF_F[6]
              
              LDM[N-1,N-1] = 1.0
              LDM[N-1,N-2] = beta
              RDM[N-1,N-1] = np.sum(CF_F[0:-1])
              RDM[N-1,N-2] = -CF_F[0]
              RDM[N-1,N-3] = -CF_F[1]
              RDM[N-1,N-4] = -CF_F[2]
              RDM[N-1,N-5] = -CF_F[3]
              RDM[N-1,N-6] = -CF_F[4]
              RDM[N-1,N-7] = -CF_F[5]
              RDM[N-1,N-8] = -CF_F[6]
       
       if end6: 
              # Coefficients for 6th order compact one-sided schemes
              hp1 = dom[1] - dom[0]
              hp2 = dom[2] - dom[1]
              hp3 = dom[3] - dom[2]
              hp4 = dom[4] - dom[3]
              hp5 = dom[5] - dom[4]
       
              # Compute the stencil coefficients
              CME = endMatrix6(hp1, hp2, hp3, hp4, hp5)
              CME_V = np.zeros(6)
              CME_V[0] = 1.0
              
              Q, R = scl.qr(CME)
              PLU = scl.lu_factor(R)
              CF_F = scl.lu_solve(PLU, (Q.T).dot(CME_V))
              beta = CF_F[-1]
              
              LDM[0,0] = 1.0
              LDM[0,1] = beta
              RDM[0,0] = -np.sum(CF_F[0:-1])
              RDM[0,1] = CF_F[0]
              RDM[0,2] = CF_F[1]
              RDM[0,3] = CF_F[2]
              RDM[0,4] = CF_F[3]
              RDM[0,5] = CF_F[4]
              
              LDM[N-1,N-1] = 1.0
              LDM[N-1,N-2] = beta
              RDM[N-1,N-1] = np.sum(CF_F[0:-1])
              RDM[N-1,N-2] = -CF_F[0]
              RDM[N-1,N-3] = -CF_F[1]
              RDM[N-1,N-4] = -CF_F[2]
              RDM[N-1,N-5] = -CF_F[3]
              RDM[N-1,N-6] = -CF_F[4]
              
       if end4: 
              # Coefficients for 6th order compact one-sided schemes
              hp1 = dom[1] - dom[0]
              hp2 = dom[2] - dom[1]
              hp3 = dom[3] - dom[2]
              hp4 = dom[4] - dom[3]
              hp5 = dom[5] - dom[4]
       
              # Compute the stencil coefficients
              CME = endMatrix4(hp1, hp2, hp3)
              CME_V = np.zeros(4)
              CME_V[0] = 1.0
              
              Q, R = scl.qr(CME)
              PLU = scl.lu_factor(R)
              CF_F = scl.lu_solve(PLU, (Q.T).dot(CME_V))
              beta = CF_F[-1]
              
              LDM[0,0] = 1.0
              LDM[0,1] = beta
              RDM[0,0] = -np.sum(CF_F[0:-1])
              RDM[0,1] = CF_F[0]
              RDM[0,2] = CF_F[1]
              RDM[0,3] = CF_F[2]
              
              LDM[N-1,N-1] = 1.0
              LDM[N-1,N-2] = beta
              RDM[N-1,N-1] = np.sum(CF_F[0:-1])
              RDM[N-1,N-2] = -CF_F[0]
              RDM[N-1,N-3] = -CF_F[1]
              RDM[N-1,N-4] = -CF_F[2]
       
       if end3:
              hp2 = dom[1] - dom[0]
              hp3 = dom[2] - dom[1]
              LDM[0,0] = 1.0
              LDM[0,1] = (hp2 + hp3) / hp3
              RDM[0,0] = -(3.0 * hp2 + 2.0 * hp3) / (hp2 * (hp2 + hp3))
              RDM[0,1] = (hp2 + hp3) * (2.0 * hp3 - hp2) / (hp2 * hp3**2)
              RDM[0,2] = (hp2**2) / (hp3**2 * (hp2 + hp3))
              '''
              '''
              hp2 = dom[N-2] - dom[N-1]
              hp3 = dom[N-3] - dom[N-2]
              LDM[N-1,N-1] = 1.0
              LDM[N-1,N-2] = (hp2 + hp3) / hp3
              RDM[N-1,N-1] = -(3.0 * hp2 + 2.0 * hp3) / (hp2 * (hp2 + hp3))
              RDM[N-1,N-2] = (hp2 + hp3) * (2.0 * hp3 - hp2) / (hp2 * hp3**2)
              RDM[N-1,N-3] = (hp2**2) / (hp3**2 * (hp2 + hp3))
              
       # Get the derivative matrix
       PLU = scl.lu_factor(LDM)
       DDM = scl.lu_solve(PLU, RDM)
       
       #DDM1 = removeLeastSingularValue(DDM)
       DDM = numericalCleanUp(DDM)
       
       return DDM

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
       
       # Make a diagonal matrix of weight
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
       
       #DDM1 = removeLeastSingularValue(DDM)
       #DDMC = numericalCleanUp(DDM)
       
       return DDM.astype(np.float64), STR_H

def computeChebyshevDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       # Initialize grid and make column vector
       xi, wcp = cheblb(NZ)
       
       b = 2.0 / ZH
   
       # Get the Chebyshev transformation matrix
       CT = chebpolym(NZ+1, xi)
       CT = numericalCleanUp(CT)
   
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
       temp = CT @ W
       STR_C = S @ temp
       # Chebyshev spatial derivative based on spectral differentiation
       # Domain scale factor included here
       temp = CT.T @ SDIFF
       DDM = b * temp @ STR_C
       
       #DDM1 = removeLeastSingularValue(DDM)
       #DDMC = numericalCleanUp(DDM)
       
       return DDM, STR_C

def computeFourierDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       L1 = DIMS[0]
       L2 = DIMS[1]
       NX = DIMS[3]
       
       kxf = (2*mt.pi/abs(L2 - L1)) * np.fft.fftfreq(NX) * NX
       KDM = np.diag(kxf, k=0)
       DFT = np.fft.fft(np.eye(NX), axis=0)
       DDM = np.fft.ifft(1j * KDM.dot(DFT), axis=0)
       
       #DDMC = numericalCleanUp(DDM)
       
       return np.real(DDM), DFT

def computeLaguerreDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       NM = NZ+1
       
       xi, wlf = lgfunclb(NZ)
       LT = lgfuncm(NM-1, xi, True)
              
       # Get the scale factor
       b = max(xi) / ZH
       
       # Make a diagonal matrix of weights
       W = np.diag(wlf, k=0)
       
       # Compute scaling for the forward transform
       S = np.eye(NM) # Identity matrix in the simplest case of Laguerre
       '''
       for ii in range(NM):
              S[ii,ii] = mt.factorial(ii) / mt.gamma(ii+1)
       '''              
       # Compute the spectral derivative coefficients
       SDIFF = np.zeros((NM,NM))
         
       for rr in range(NM):
              SDIFF[rr,rr] = 0.5
              for cc in range(rr,NM):
                     SDIFF[rr,cc] -= 1.0
       # Hermite function spectral transform in matrix form
       temp = (LT).dot(W)
       STR_L = S.dot(temp)
       # Hermite function spatial derivative based on spectral differentiation
       temp = (LT.T).dot(SDIFF)
       temp = temp.dot(STR_L)       
       DDM = b * temp
       
       #DDM1 = removeLeastSingularValue(DDM)
       #DDMC = numericalCleanUp(DDM)
       
       return DDM.astype(np.float64), STR_L

def computeLegendreDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       xi, wlf = leglb(NZ)
       LT, DLT = legpolym(NZ, xi, True)
       #LT = numericalCleanUp(LT)
       
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
       temp = LT @ W
       STR_L = S @ temp
       # Legendre spatial derivative based on spectral differentiation
       # Domain scale factor included here
       temp = LT.T @ SDIFF
       DDM = b * temp @ STR_L
       
       #DDM1 = removeLeastSingularValue(DDM)
       DDMC = numericalCleanUp(DDM)
       
       return DDM, STR_L
       
       