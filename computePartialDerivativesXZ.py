#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:10:18 2019

@author: TempestGuerra
"""

import numpy as np
import scipy.sparse as sps

def computePartialDerivativesXZ(DIMS, REFS, DDX_1D, DDZ_1D):
       # Get the dimensions
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Get REFS data
       sigma = REFS[7]
       
       # Unwrap the 1D derivative matrices into 2D operators
       
       #%% Vertical derivative and diffusion operators
       DDZ_OP = np.empty((OPS,OPS))
       for cc in range(NX):
              # Advanced slicing used to get submatrix
              ddex = np.array(range(NZ)) + cc * NZ
              # TF adjustment for vertical coordinate transformation
              SIGMA = sps.diags(sigma[:,cc])
              DDZ_OP[np.ix_(ddex,ddex)] = SIGMA.dot(DDZ_1D)
              
       # Make the operators sparse
       DDZM = sps.csr_matrix(DDZ_OP); del(DDZ_OP)
       
       '''#%% Vertical viscous operator
       VSZ_OP = np.empty((OPS,OPS))
       for cc in range(NX):
              # Advanced slicing used to get submatrix
              ddex = np.array(range(NZ)) + cc * NZ
              # TF adjustment for vertical coordinate transformation
              SIGMA = sps.diags(sigma[:,cc])
              VSZ_OP[np.ix_(ddex,ddex)] = SIGMA.dot(VSZ_1D)
              
       # Make the operators sparse
       VSZM = sps.csr_matrix(VSZ_OP); del(VSZ_OP)
       '''              
       #%% Horizontal Derivative
       DDX_OP = np.empty((OPS,OPS))
       for rr in range(NZ):
              ddex = np.array(range(0,OPS,NZ)) + rr
              # Advanced slicing used to get submatrix
              DDX_OP[np.ix_(ddex,ddex)] = DDX_1D
              
       # Make the operators sparse
       DDXM = sps.csr_matrix(DDX_OP); del(DDX_OP)

       '''#%% Horizontal viscous operators
       VSX_OP = np.empty((OPS,OPS))
       for rr in range(NZ):
              ddex = np.array(range(0,OPS,NZ)) + rr
              # Compute the viscous operator
              VSX_OP[np.ix_(ddex,ddex)] = VSX_1D
              
       # Make the operators sparse
       VSXM = sps.csr_matrix(VSX_OP); del(VSX_OP)
       '''
       return DDXM, DDZM#, VSXM, VSZM