#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:10:56 2019

@author: -
"""

import numpy as np
#import scipy.interpolate as spint
import HerfunChebNodesWeights as hcnw

def computeColumnInterp(DIMS, zdata, fdata, ZTL, ITRANS):
       NX = ZTL.shape[1]
       NZ = fdata.shape[0]
       
       # Check that data is good for self interpolation
       if (zdata.all() == None) or (fdata.all() == None):
              print('ERROR: No reference data for interpolation given!')
              FLD = np.tile(fdata, NX)
              return FLD
       
       # Interpolated field is of the same size as the original
       FLDI = np.zeros(ZTL.shape)
       # Apply forward transform on the nominal column
       fcoeffs = ITRANS.dot(fdata)
                     
       # Loop over each column
       for cc in range(NX):
              # Convert to the reference grid at this column
              thisZ = ZTL[:,cc]
              xi = ((2.0 / np.amax(thisZ) * thisZ) - 1.0)
              
              # Apply the interpolation
              ITM, dummy = hcnw.legpolym(NZ-1, xi, True)
              temp = (ITM.T).dot(fcoeffs)
              
              FLDI[:,cc] = np.ravel(temp)
              
       return FLDI
              