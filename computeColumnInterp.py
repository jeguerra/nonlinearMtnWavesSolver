#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:10:56 2019

@author: -
"""

import numpy as np
#import scipy.interpolate as spint
import HerfunChebNodesWeights as hcnw

def computeColumnInterp(DIMS, zdata, fdata, NZI, ZTL, FLD, CH_TRANS, TypeInt):
       NX = DIMS[3]
       NZ = DIMS[4]
       
       # Interpolate the nominal column profile to TF Chebyshev grid
       if TypeInt == '1DtoTerrainFollowingCheb':
              # Check that data is good for self interpolation
              if (zdata.all() == None) or (fdata.all() == None):
                     print('ERROR: No reference data for interpolation given!')
                     return FLD
              
              # Interpolated field is of the same size as the original
              FLDI = np.zeros((NZ, NX))
              # Compute the total height of nominal column
              zpan = np.amax(zdata) - np.min(zdata)
              # Apply forward transform on the nominal column
              fcoeffs = CH_TRANS.dot(fdata)
              
              # Loop over each column
              for cc in range(NX):
                     # Convert to the reference grid at this column
                     thisZ = ZTL[:,cc]
                     xi = 1.0 * ((2.0 / zpan * thisZ) - 1.0)
                     
                     # Get the Chebyshev matrix for this column
                     CTM = hcnw.chebpolym(NZ-1, -xi)
                     
                     # Apply the interpolation
                     temp = (CTM).dot(fcoeffs)
                     FLDI[:,cc] = np.ravel(temp)
                     
              return FLDI
                     
       # Interpolate solution on TF Chebyshev grid to TF linear grid
       elif TypeInt == 'TerrainFollowingCheb2Lin':
              # Check
              if NZI <= 0:
                     print('ERROR: Invalid number of points in new grid! ', NZI)
                     return FLD
              
              # Interpolated field has a new size
              ZTLI = np.zeros((NZI, NX))
              FLDI = np.zeros((NZI, NX))
              # Compute the new column reference grid (linear space)
              xi = np.linspace(-1.0, 1.0, num=NZI, endpoint=True)
              
              # Loop over each column
              for cc in range(NX):
                     # Compute the new column physical grid
                     zdata = ZTL[:,cc]
                     zmin = np.min(zdata)
                     zmax = np.amax(zdata)
                     zspan = zmax - zmin
                     zi = zspan * 0.5 * xi + (0.5 * zmax) + (0.5 * zmin)
                     ZTLI[:,cc] = np.ravel(zi)
                     
                     # Apply the forward transform at this column
                     fcoeffs = CH_TRANS.dot(FLD[:,cc])
                     
                     # Get the Chebyshev matrix for this column
                     CTM = hcnw.chebpolym(NZ-1, -xi)
                     
                     # Apply the interpolation
                     temp = (CTM).dot(fcoeffs)
                     FLDI[:,cc] = np.ravel(temp)
              
              return FLDI, ZTLI
              