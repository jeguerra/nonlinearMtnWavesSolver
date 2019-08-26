#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:05:23 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
from HerfunChebNodesWeights import chebpolym, cheblb

def computeChebyshevDerivativeMatrix(DIMS):
       
       # Get data from DIMS
       ZH = DIMS[2]
       NZ = DIMS[4]
       
       # Make a truncation index
       tdex = np.array(range(NZ-10,NZ), dtype=int)
       
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
       temp = (CTD[:,tdex].T).dot(W)
       STR_C = S[np.ix_(tdex,tdex)].dot(temp);
       # Chebyshev spatial derivative based on spectral differentiation
       # Domain scale factor included here
       temp = (CTD[:,tdex]).dot(SDIFF[np.ix_(tdex,tdex)])
       DDM = - (2.0 / ZH) * temp.dot(STR_C);
       
       return DDM, STR_C