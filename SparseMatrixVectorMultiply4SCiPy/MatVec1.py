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
import torch
import scipy.sparse as sps
from joblib import Parallel, delayed
from rsb import rsb_matrix

import math as mt
import matplotlib.pyplot as plt
from computeGrid import computeGrid
import computeDerivativeMatrix as derv
import computeTopographyOnGrid as top
                     
if __name__ == '__main__':
       
        # Set grid dimensions and order
       L2 = 1.0E4 * 3.0 * mt.pi
       L1 = -L2
       ZH = 36000.0
       NX = 583
       NZ = 92
       DIMS = [L1, L2, ZH, NX, NZ]
       
       # Define the computational and physical grids+
       REFS = computeGrid(DIMS, True, False, True, False)
       
       #% Compute the raw derivative matrix operators in alpha-xi computational space
       A, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       A = A.astype(float)
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
       
       # Test native sparse pytorch dot product
       APT = torch.tensor(A)
       APST = APT._to_sparse_csr()
       start = time.time()
       R2 = torch.sparse.mm(APST, torch.tensor(V))
       end = time.time()
       print('PyTorch SpMV: ', end - start, ' sec')
       
       # Test SpMV serial implementation
       start = time.time()
       R3 = ACSR.dot(V)
       end = time.time()
       print('SciPy SpMV: ', end - start, ' sec')