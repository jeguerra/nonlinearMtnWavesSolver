#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:07:48 2019

@author: TempestGuerra
"""

import numpy as np
from numpy import multiply as mul
import math as mt
from scipy.special import roots_hermite
from scipy.special import roots_genlaguerre
from scipy.special import roots_legendre
#from scipy import linalg as las

def lgfunclb(NX):
       
       # Compute gauss-laguerre nodes and weights
       xi, dum = roots_genlaguerre(NX, 1)
       
       xir = np.zeros(NX+1)       
       for ii in range(1,len(xir)):
              xir[ii] = xi[ii-1]
       
       # Compute the Hermite function weights
       lf = lgfuncm(NX, xir, False)
       w = 1.0 / (NX+1) * np.power(lf, -2.0, dtype=np.longdouble)

       return xir, w

def lgfuncm(NX, xi, fullMat):
       
       # Initialize constant
       ND = len(xi)
       
       #'''
       # Initialize the output matrix if needed
       if fullMat:
              LFM = np.zeros((NX+1,ND))
              
       # Compute the first two modes of the recursion
       wfun = np.exp(-0.5 * xi, dtype=np.longdouble)
       poly0 = wfun
       poly1 = wfun * (1.0 - xi)
       
       # Put the first two functions in the matrix or return low order functions
       if fullMat:
              LFM[0,:] = poly0
              LFM[1,:] = poly1
       elif NX == 0:
              return poly0
       elif NX == 1:
              return poly1
       
       for nn in range(2,NX+1):
              polyn = (2 * (nn - 1) - xi + 1) / nn * poly1
              polyn -= (nn - 1) / nn * poly0
              poly0 = poly1; 
              poly1 = polyn;
              # Put the new function in its matrix place
              if fullMat:
                     LFM[nn,:] = polyn
              else:
                     LFM = polyn
       #'''
       return LFM

def hefunclb(NX):
       #'''
       # Compute gauss-hermite nodes and weights
       xi, dum = roots_hermite(NX+1)
       # Compute the Hermite function weights
       hf = hefuncm(NX, xi, False)
       w = 1.0 / (NX+1) * np.power(hf, -2.0, dtype=np.longdouble)
       #'''     
       return xi, w
       
def hefuncm(NX, xi, fullMat):
       # Initialize constant
       cst = mt.pi**(-0.25)
       ND = len(xi)
       
       #'''
       # Initialize the output matrix if needed
       if fullMat:
              HFM = np.zeros((NX+1,ND))
              
       # Compute the first two modes of the recursion
       x2 = np.power(xi, 2.0, dtype=np.longdouble)
       wfun = np.exp(-0.5 * x2, dtype=np.longdouble)
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
              poly0 = poly1
              poly1 = polyn
              # Put the new function in its matrix place
              if fullMat:
                     HFM[nn+1,:] = polyn
              else:
                     HFM = polyn
       #'''
       return HFM

def cheblb(NZ):
       # Compute Chebyshev CGL nodes and weights
       xc = np.array(range(NZ+1))
       xi = -np.cos(mt.pi / NZ * xc)
       
       w = mt.pi / NZ * np.ones(NZ+1)
       w[0] *= 0.5
       w[NZ] *= 0.5
       
       return xi, w
   
def chebpolym(NM, xi):
       # Compute Chebyshev pols (first kind) into a matrix transformation
       # Functions need to be arranged bottom to top!
       NX = len(xi)
       CTM = np.zeros((NX, NM))
       
       CTM[:,0] = np.ones(NX)
       CTM[:,1] = xi
       
       # 3 Term recursion for functions
       for ii in range(2, NM):
              CTM[:,ii] = 2.0 * \
              mul(xi, CTM[:,ii-1]) - \
              CTM[:,ii-2]
              
       return CTM

def leglb(NZ):
       # Compute LG nodes as initial guess
       xlz, wl = roots_legendre(NZ)
       xi = np.zeros(NZ+1)
       xi[0] = -1.0
       xi[-1] = 1.0
       
       #print('Zeros of LP:', xlz)
       
       NI = 100
       kk = 1
       # Compute interior zeros of DLP for LGL
       for ii in range(1,len(xlz)):
              # Initialize to between zeros of LP
              xl = 0.5 * (xlz[ii] + xlz[ii-1])
              # Loop over Newton iterations
              for nn in range(NI):
                     LN, DLN = legpolym(NZ, xl, False)
                     xl -= (1.0 - xl**2) * DLN / (2.0 * xl * DLN - NZ * (NZ + 1) * LN)

              xi[kk] = xl
              kk += 1
                   
       #print('Zeros of DLP:', xi)
       LN, DLN = legpolym(NZ, xi, False)
       #print((1.0-np.power(xi,2.0)) * DLN)
       
       # Compute the weights
       wl = 2.0 / (NZ * (NZ + 1)) * np.power(LN, -2.0)
       
       return xi, wl

def legpolym(ND, xi, fullMat):
       
       try:
              NX = len(xi)
       except:
              NX = 1
       
       # Initialize the polynomials and their derivatives
       poly0 = np.ones(NX)
       poly1 = xi
       dpoly0 = np.zeros(NX)
       dpoly1 = np.ones(NX)
                        
       # Initialize the output matrices if needed
       if fullMat:
              LTM = np.zeros((ND+1,NX))
              LTM[0,:] = poly0
              LTM[1,:] = poly1
              DLTM = np.zeros((ND+1,NX))
              DLTM[0,:] = dpoly0
              DLTM[1,:] = dpoly1
       
       for nn in range(2,ND+1):
              # Compute the new polynomial
              polyn = (2 * (nn-1) + 1) / nn * xi * poly1
              polyn -= (nn-1) / nn * poly0
              # Compute the new polynomial derivative
              dpolyn = (2 * (nn-1) + 1) * poly1 + dpoly0
              # Update polynomials
              poly0 = poly1
              poly1 = polyn
              # Update polynomial derivatives
              dpoly0 = dpoly1
              dpoly1 = dpolyn
              # Put the new function in its matrix place
              if fullMat:
                     LTM[nn,:] = polyn
                     DLTM[nn,:] = dpolyn
              else:
                     LTM = polyn
                     DLTM = dpolyn
       
       return LTM, DLTM