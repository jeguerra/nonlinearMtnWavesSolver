#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 08:39:12 2019

@author: TempestGuerra
"""

import numpy as np
import scipy.linalg as dsl

def computeSchurSolve(SYS, b):
       # Get the shape of A
       M = SYS.shape[0]
       N = SYS.shape[1]
       MF = len(b)
       
       # Check sizes...
       if M != N:
              print('NOT A SQUARE MATRIX IN SCHUR SOLVER!')
              return b
       elif MF != M:
              print('FORCING VECTOR LENGTH NOT EQUAL TO MATRIX ROWS!')
              return b
       else:
              print('Solving sub-matrix problem with Schur Complement...')
              
       # Partition the system
       if M % 2 == 0:
              M1 = int(M / 2)
              #M2 = M1
       else:
              M1 = int((M - 1) / 2)
              #M2 = M - M1
       
       # Compute partition indices
       idex1 = np.array(range(0,M1))
       idex2 = np.array(range(M1,M))
       
       # Set the 4 blocks and forcing
       A = SYS[np.ix_(idex1, idex1)]
       B = SYS[np.ix_(idex1, idex2)]
       C = SYS[np.ix_(idex2, idex1)]
       D = SYS[np.ix_(idex2, idex2)]
       f1 = b[idex1]
       f2 = b[idex2]
       
       # Get some memory back
       del(SYS)
       del(b)
       
       # Factor D
       factorD = dsl.lu_factor(D)
       del(D)
       # Solve D against C
       alpha = dsl.lu_solve(factorD, C)
       # Compute alpha f2_hat = D^-1 * f2 and f1_hat
       f2_hat = dsl.lu_solve(factorD, f2)
       f1_hat = f1 - B.dot(f2_hat)
       del(f2_hat)
       # Compute Schur Complement of D
       D_SC = A - B.dot(alpha)
       del(A)
       del(B)
       del(alpha)
       # Factor Schur Complement of D
       factorD_SC = dsl.lu_factor(D_SC)
       del(D_SC)
       # Compute the solution
       sol1 = dsl.lu_solve(factorD_SC, f1_hat)
       del(factorD_SC)
       f2_hat = f2 - C.dot(sol1)
       del(C)
       sol2 = dsl.lu_solve(factorD, f2_hat)
       del(factorD)
       dsol = np.concatenate((sol1, sol2))
       
       return dsol