#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:49:13 2020

CSR sparse matrix vector multiplication 
Multithreading acceleration test

@author: jeguerra
"""
import time
import numpy as np
import scipy.sparse as sps
from joblib import Parallel, delayed
from oct2py import Oct2Py
from rsb import rsb_matrix

import math as mt
import matplotlib.pyplot as plt
from computeGrid import computeGrid
import computeDerivativeMatrix as derv
import computeTopographyOnGrid as top
                     
def computeMatVecDotParfor(aCSRmat, aDenseVec, runParFor):
       
       N = aCSRmat.shape[0]
       resVec = np.zeros((N,))
       
       def innerLoop(ii):
              c1 = aCSRmat.indptr[ii]
              c2 = aCSRmat.indptr[ii+1]
              cdex = np.arange(c1,c2)
              
              # vectorized inner loop
              res = (aCSRmat.data[cdex]).dot(aDenseVec[aCSRmat.indices[cdex]])
              
              #for cc in range(c1,c2):
              #       res += aCSRmat.data[cc] * aDenseVec[aCSRmat.indices[cc]]
              
              return res
                            
       if runParFor:
              resLst = Parallel(n_jobs=4, prefer='threads')(delayed(innerLoop)(rr) for rr in range(N))
              resVec = np.array(resLst)
       else:
              for rr in range(N):
                     resVec[rr] = innerLoop(rr)

       return resVec

if __name__ == '__main__':
       
        # Set grid dimensions and order
       L2 = 1.0E4 * 3.0 * mt.pi
       L1 = -L2
       ZH = 36000.0
       NX = 347
       NZ = 92
       DIMS = [L1, L2, ZH, NX, NZ]
       
       # Define the computational and physical grids+
       REFS = computeGrid(DIMS, True, False, True)
       
       #% Compute the raw derivative matrix operators in alpha-xi computational space
       A, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       #A, HF_TRANS = derv.computeFourierDerivativeMatrix(DIMS)
       #A = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[0])
       #A = derv.computeCubicSplineDerivativeMatrix(DIMS, REFS[0], True)
       
       # Make a dense teste vector (MUST be NX1 for Octave to work)
       ACSR = sps.csr_matrix(np.real(A))
       V = np.expand_dims(REFS[0],1) * np.ones((NX+1,1))
       
       # Test native dot product
       start = time.time()
       R1 = A.dot(V)
       end = time.time()
       print('Native numpy MV: ', end - start, ' sec')
       
       # Test native rsb dot product
       AR = rsb_matrix(ACSR)
       #AR.autotune()
       start = time.time()
       R2 = AR.dot(V)
       end = time.time()
       print('PyRSB SpMV: ', end - start, ' sec')
       
       # Test SpMV serial implementation
       start = time.time()
       R3 = computeMatVecDotParfor(ACSR, V, False)
       end = time.time()
       print('Serial SpMV: ', end - start, ' sec')
       
       # Test SpMV parallel implementation
       start = time.time()
       R4 = computeMatVecDotParfor(ACSR, V, True)
       end = time.time()
       print('Parallel SpMV: ', end - start, ' sec')
       
       # Test SpMV from Octave bridge
       oc = Oct2Py(temp_dir='/dev/shm/')
       start = time.time()
       R5 = oc.mtimes(A,V)
       end = time.time()
       print('Octave SpMV: ', end - start, ' sec')
       
