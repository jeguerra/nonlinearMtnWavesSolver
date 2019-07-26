#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:10:56 2019

@author: -
"""

import numpy as np
import scipy.interpolate as spint
import HerfunChebNodesWeights as hcnw

def chebyshev_coef_1d ( nd, xd, yd ):

#*****************************************************************************80
#
## CHEBYSHEV_COEF_1D determines the Chebyshev interpolant coefficients.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    26 July 2017
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer ND, the number of data points.
#    ND must be at least 1.
#
#    Input, real XD(ND), the data locations.
#
#    Input, real YD(ND), the data values.
#
#    Output, real C(ND), the Chebyshev coefficients.
#
#    Output, real XMIN, XMAX, the interpolation interval.
#
  import numpy as np

  if ( nd == 1 ):
    c = 1.0
    xmin = xd[0]
    xmax = xd[0]
    return c, xmin, xmax

  xmin = min ( xd )
  xmax = max ( xd )
#
#  Map XD to [-1,+1].
#
  x = ( 2.0 * xd - xmin - xmax ) / ( xmax - xmin )
#
#  Form the Chebyshev Vandermonde matrix.
#
  a = np.outer ( np.arccos ( x ), np.arange ( 0, nd ) )
  a = np.cos ( a )
#
#  Solve for the expansion coefficients.
#
  c = np.linalg.solve ( a, yd )

  return c, xmin, xmax

def computeColumnInterp(DIMS, zdata, fdata, ZTL, FLD, CH_TRANS):
       NX = DIMS[3]
       NZ = DIMS[4]
       
       # Apply forward transform on the nominal column
       fcoeffs1 = CH_TRANS.dot(fdata)
       #fcoeffs2, zmin, zmax = chebyshev_coef_1d(NZ, zdata, fdata)
       #plt.figure()
       #plt.plot(np.ravel(fcoeffs1))
       #plt.plot(np.ravel(fcoeffs2))
       #splint = spint.interp1d(zdata, fdata.T, kind='cubic')
       
       # Loop over each column
       for cc in range(NX):
              # Convert to the reference grid at this column
              thisZ = ZTL[:,cc]
              #'''
              zpan = np.amax(zdata) - np.min(zdata)
              xi = 1.0 * ((2.0 / zpan * thisZ) - 1.0)
              # Get the Chebyshev matrix for this column
              CTM = hcnw.chebpolym(NZ-1, -xi)
              # Apply the interpolation
              FLDI = (CTM).dot(fcoeffs1)
              #'''
              # PUNT... CUBIC SPLINE INTERPOLANT
              #FLDI = splint(thisZ)
              FLD[:,cc] = np.ravel(FLDI)
              
       return FLD
              