#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:10:18 2019

@author: TempestGuerra
"""

import numpy as np
import scipy.sparse as sps

def computePartialDerivativesXZ(DIMS, REFS, dzdh, DDX_1D, DDZ_1D):
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Get REFS data
       DZT = REFS[6]
       sigma = REFS[7]
       
       # Unwrap the 1D derivative matrices into 2D operators
       
       # Vertical derivative
       DDZ_OP = np.empty((OPS,OPS))
       for cc in range(NX):
              ddex = np.array(range(NZ)) + cc * NZ
              # Advanced slicing used to get submatrix
              DDZ_OP[np.ix_(ddex,ddex)] = DDZ_1D
       
       
       # Horizontal Derivative
       DDX_OP = np.empty((OPS,OPS))
       for rr in range(NZ):
              ddex = np.array(range(0,OPS,NZ)) + rr
              # Advanced slicing used to get submatrix
              DDX_OP[np.ix_(ddex,ddex)] = DDX_1D
              
       #%% Make the operators sparse
       DDXM = sps.csc_matrix(DDX_OP)
       DDZM = sps.csc_matrix(DDZ_OP)
       
       # Apply terrain following adjustments
       SIGMA = sps.diags(np.reshape(sigma, (OPS,), order='F'), format='csc')
       #DZDX = sps.diags(np.reshape(DZT, (OPS,), order='F'), format='csc')
       
       DDZM = SIGMA.dot(DDZM)
       #DDXM = DDXM + DZDX.dot(DDZM)
       
       return DDXM, DDZM