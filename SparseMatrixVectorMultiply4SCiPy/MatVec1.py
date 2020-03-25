#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:49:13 2020

CSR sparse matrix vector multiplication 
Multithreading acceleration test

@author: jeguerra
"""
import numpy as np
import scipy.sparse as sps
from joblib import Parallel, delayed

def computeMatVecDot(aCSRmat, aDenseVec):
       
       N = aCSRmat.shape[0]
       resVec = np.zeros((N,))
       
       for rr in range(N):
              c1 = aCSRmat.indptr[rr]
              c2 = aCSRmat.indptr[rr+1]
              for cc in range(c1,c2):
                     resVec[rr] = aCSRmat.data[cc] * aDenseVec[aCSRmat.indices[cc]]
                     
       return resVec
                     
def computeMatVecDotParfor(aCSRmat, aDenseVec, runParFor):
       
       N = aCSRmat.shape[0]
       resVec = np.zeros((N,))
       
       def innerLoop(ii):
              c1 = aCSRmat.indptr[ii]
              c2 = aCSRmat.indptr[ii+1]
              for cc in range(c1,c2):
                     thisRes = aCSRmat.data[cc] * aDenseVec[aCSRmat.indices[cc]]
                     
              return thisRes
       
       if runParFor:
              Parallel(n_jobs=2)(delayed(innerLoop)(rr) for rr in range(N))
       else:
              for rr in range(N):
                     resVec[rr] = innerLoop[rr]

       return resVec
