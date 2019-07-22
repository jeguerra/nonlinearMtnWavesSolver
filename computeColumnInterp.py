#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:10:56 2019

@author: -
"""

import numpy as np
import HerfunChebNodesWeights as hcnw

def computeColumnInterp(DIMS, ZTL, FLD, CH_TRANS):
       ZH = DIMS[2]
       NX = DIMS[3]
       NZ = DIMS[4]
       
       # Apply forward transform on the nominal column
       fcoeffs = np.matmul(CH_TRANS, ZTL[:,0])
       
       # Loop over each column
       for cc in range(NX):
              # Convert to the reference grid at this column
              xi = (1.0 - 2.0 / ZH * ZTL[:,cc])
              # Get the Chebyshev matrix for this column
              CTM = hcnw.chebpolym(NZ, xi)
              # Apply the interpolation
              FLD[:,cc] = np.matmul(CTM, fcoeffs)
              
       return FLD
              