#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:10:18 2019

@author: TempestGuerra
"""

import numpy as np
import scipy.sparse as sps

def computePartialDerivativesXZ(DIMS, REFS):
       # Get the dimensions
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Get REFS data
       DDX_1D = REFS[2]
       DDZ_1D = REFS[3]
       sigma = REFS[7]
       
       # Unwrap the 1D derivative matrices into 2D operators
       
       # Vertical derivative
       DDZ_OP = np.empty((OPS,OPS))
       for cc in range(NX):
              # Compute the terrain adjusment for this column
              terrainFollowScale = np.diag(sigma[:,cc], k=0)
              DDZ_TF1D = np.matmul(terrainFollowScale, DDZ_1D)   
              ddex = np.array(range(NZ)) + cc * NZ
              # Advanced slicing used to get submatrix
              DDZ_OP[np.ix_(ddex,ddex)] = DDZ_TF1D
       
       
       # Horizontal Derivative
       DDX_OP = np.empty((OPS,OPS))
       for rr in range(NZ):
              ddex = np.array(range(0,OPS,NZ)) + rr
              # Advanced slicing used to get submatrix
              DDX_OP[np.ix_(ddex,ddex)] = DDX_1D
              
       #%% Make the operators sparse
       DDXM = sps.csc_matrix(DDX_OP)
       DDZM = sps.csc_matrix(DDZ_OP)
       
       return DDXM, DDZM