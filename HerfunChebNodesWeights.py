#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:07:48 2019

@author: TempestGuerra
"""

import numpy as np
from numpy import multiply as mul
from scipy import linalg as las
import math as mt
from scipy.special import roots_hermite
from scipy.special import roots_chebyt

def hefunclb(NX):
       #'''
       # Compute off-diagonals of 7.84 in Spectral Methods, Springer
       b = range(1,NX+1)
       bd = 0.5 * np.array(b)
       
       # Assemble the matrix
       m1 = np.diag(np.sqrt(bd), k=+1)
       m2 = np.diag(np.sqrt(bd), k=-1)
       mm = np.add(m1,m2)
       
       # Compute the eigenvalues of this matrix (zeros Hermite polys)
       ew = las.eigvals(mm)
       # Sort the eigenvalues in ascending order and store nodes
       xi = np.sort(np.real(ew))
       
       # Compute the Hermite function weights
       hf = hefuncm(NX, xi, False)
       w = 1.0 / (NX+1) * np.power(hf, -2.0)
       #'''
       '''
       xi, w = roots_hermite(NX+1)
       
       # Compute the Hermite function weights
       hf = hefuncm(NX, xi, False)
       w = 1.0 / (NX+1) * np.power(hf, -2.0)
       '''     
       return xi, w
       
def hefuncm(NX, xi, fullMat):
       # Initialize constant
       cst = 1.0 / mt.pi**4;
       ND = len(xi)
       
       # Initialize the output matrix if needed
       if fullMat:
              HFM = np.zeros((NX+1,ND))
              
       # Compute the first two modes of the recursion
       wfun = np.exp(-0.5 * np.power(xi, 2.0))
       poly0 = cst * wfun;
       poly1 = cst * mt.sqrt(2.0) * (xi * wfun);
       
       # Put the first two functions in the matrix or return low order functions
       if fullMat:
              HFM[0,:] = poly0
              HFM[1,:] = poly1
       elif NX == 0:
              return poly0
       elif NX == 1:
              return poly1
       
       for nn in range(1,NX):
              polyn = mt.sqrt(2.0 / (nn+1)) * (xi * poly1)
              polyn -= mt.sqrt(nn / (nn+1)) * poly0
              poly0 = poly1; 
              poly1 = polyn;
              # Put the new function in its matrix place
              if fullMat:
                     HFM[nn+1,:] = polyn
              else:
                     HFM = polyn
       
       return HFM.T

def cheblb(NZ):
       # Compute Chebyshev CGL nodes and weights
       ep = NZ - 1
       xc = np.array(range(NZ))
       xi = -np.cos(mt.pi / ep * xc)
       
       w = mt.pi / NZ * np.ones(NZ)
       w[0] *= 0.5
       w[ep] *= 0.5
       
       return xi, w
   
def chebpolym(NM, xi):
       # Compute Chebyshev pols (first kind) into a matrix transformation
       # Functions need to be arranged bottom to top!
       NX = len(xi)
       CTM = np.zeros((NX, NM+1))
       
       CTM[:,0] = np.ones(NX)
       CTM[:,1] = xi
       
       # 3 Term recursion for functions
       for ii in range(2, NM+1):
              CTM[:,ii] = 2.0 * \
              mul(xi, CTM[:,ii-1]) - \
              CTM[:,ii-2]
              
       return CTM