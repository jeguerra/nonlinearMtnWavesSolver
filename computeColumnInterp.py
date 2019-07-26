#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:10:56 2019

@author: -
"""

import numpy as np
import scipy.interpolate as spint
import HerfunChebNodesWeights as hcnw

def computeColumnInterp(DIMS, ZTL, FLD, CH_TRANS):
       ZH = DIMS[2]
       NX = DIMS[3]
       NZ = DIMS[4]
       
       # Apply forward transform on the nominal column
       #bdex = list(range(NZ-1,-1,-1))
       fcoeffs = CH_TRANS.dot(FLD[:,0])
       splint = spint.interp1d(ZTL[:,0], FLD[:,0], kind='cubic')
       
       # Loop over each column
       for cc in range(NX):
              # Convert to the reference grid at this column
              xi = -1.0 * ((2.0 / ZH * ZTL[:,cc]) - 1.0)
              # Get the Chebyshev matrix for this column
              CTM = hcnw.chebpolym(NZ, xi)
              # Apply the interpolation
              FLDI = (CTM).dot(fcoeffs)
              
              # PUNT... CUBIC SPLINE INTERPOLANT
              #FLDI = splint(ZTL[:,cc])
              FLD[:,cc] = FLDI
              
       return FLD
              