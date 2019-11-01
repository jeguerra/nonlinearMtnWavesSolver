#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:59:22 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
from HerfunChebNodesWeights import hefuncm, hefunclb
from HerfunChebNodesWeights import chebpolym, cheblb

# Computes standard 4th order compact finite difference matrix
def computeCompactFiniteDiffDerivativeMatrix(DIMS, dom):
       # Initialize the left and right derivative matrices
       N = len(dom)
       LDM = np.zeros((N,N)) # tridiagonal
       RDM = np.zeros((N,N)) # centered difference
       
       # Loop over each interior point in the irregular grid
       for ii in range(1,N-1):
              # Get the metric weights
              hp = abs(dom[ii+1] - dom[ii])
              hm = abs(dom[ii] - dom[ii-1])
              # Write the right equation
              
              # Write the left equation

def computeHermiteFunctionDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       L1 = DIMS[0]
       L2 = DIMS[1]
       NX = DIMS[3]
       
       alpha, whf = hefunclb(NX+1)
       #print(alpha)
       #print(whf)
       HT = hefuncm(NX, alpha, True)
       HTD = hefuncm(NX+1, alpha, True)
       
       # Get the scale factor
       b = (np.amax(alpha) - np.min(alpha)) / abs(L2 - L1)
       
       # Make a diagonal matrix of weights
       W = np.diag(whf, k=0)
       
       # Compute the coefficients of spectral derivative in matrix form
       SDIFF = np.zeros((NX+2,NX+1));
       SDIFF[0,1] = mt.sqrt(0.5)
       SDIFF[NX,NX-1] = -mt.sqrt(NX * 0.5);
       SDIFF[NX+1,NX] = -mt.sqrt((NX + 1) * 0.5);
                     
       for rr in range(1,NX):
              SDIFF[rr,rr+1] = mt.sqrt((rr + 1) * 0.5);
              SDIFF[rr,rr-1] = -mt.sqrt(rr * 0.5);
              
       # Hermite function spectral transform in matrix form
       STR_H = (HT.T).dot(W);
       # Hermite function spatial derivative based on spectral differentiation
       temp = (HTD).dot(SDIFF)
       temp = temp.dot(STR_H)
       DDM = b * temp
       
       return DDM, STR_H

def computeChebyshevDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       # Initialize grid and make column vector
       xi, wcp = cheblb(NZ)
   
       # Get the Chebyshev transformation matrix
       CTD = chebpolym(NZ-1, -xi)
   
       # Make a diagonal matrix of weights
       W = np.diag(wcp)
   
       # Compute scaling for the forward transform
       S = np.eye(NZ)
   
       for ii in range(NZ - 1):
              temp = W.dot(CTD[:,ii])
              temp = ((CTD[:,ii]).T).dot(temp)
              S[ii,ii] = temp ** (-1)

       S[NZ-1,NZ-1] = 1.0 / mt.pi
   
       # Compute the spectral derivative coefficients
       SDIFF = np.zeros((NZ,NZ))
       SDIFF[NZ-2,NZ-1] = 2.0 * NZ
   
       for ii in reversed(range(NZ - 2)):
              A = 2.0 * (ii + 1)
              B = 1.0
              if ii > 0:
                     c = 1.0
              else:
                     c = 2.0
            
              SDIFF[ii,:] = B / c * SDIFF[ii+2,:]
              SDIFF[ii,ii+1] = A / c
    
       # Chebyshev spectral transform in matrix form
       temp = CTD.dot(W)
       STR_C = S.dot(temp);
       # Chebyshev spatial derivative based on spectral differentiation
       # Domain scale factor included here
       temp = (CTD).dot(SDIFF)
       DDM = - (2.0 / ZH) * temp.dot(STR_C);
       
       return DDM, STR_C