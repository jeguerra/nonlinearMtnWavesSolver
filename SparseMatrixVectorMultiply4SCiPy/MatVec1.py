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
       
       # Set up problem
       N = 2048
       
       # Make a test sparse matrix
       A = sps.random(N, N, density=0.5, format='csr')
       
       # Make a dense teste vector
       V = np.ones(N)
       
       # Test native dot product
       start = time.time()
       R1 = A.dot(V)
       end = time.time()
       print('Native SpNV: ', end - start, ' sec')
       
       # Test SpNV serial implementation
       start = time.time()
       R2 = computeMatVecDotParfor(A, V, False)
       end = time.time()
       print('Serial SpNV: ', end - start, ' sec')
       
       # Test SpNV parallel implementation
       start = time.time()
       R3 = computeMatVecDotParfor(A, V, True)
       end = time.time()
       print('Parallel SpNV: ', end - start, ' sec')
       
