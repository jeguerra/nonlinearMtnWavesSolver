#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:23:58 2019

Grid stretching that preserves the layout of Chebyshev nodes on columns.

@author: TempestGuerra
"""

import numpy as np

def computeStretchedDomain2D(DIMS, REFS, zRay, hx, dhdx):
       # Get data from DIMS and REFS
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       
       # Get REFS data
       x = REFS[0]
       z = REFS[1]
       DDX_1D = REFS[2]
       
       # Compute the flat XZ mesh
       DZT, dummy = np.meshgrid(dhdx,z);
       XL, ZL = np.meshgrid(x,z);
       
       # Make the global array of terrain height and slope features
       ZTL = np.zeros((NZ,NX))
       
       sigma = np.ones((NZ,NX))
       for cc in range(NX):
              thisZH = ZH - hx[cc]
              sigma[:,cc] *= (ZH / thisZH)
              ZTL[:,cc] = ZL[:,cc] * thisZH / ZH
              ZTL[:,cc] += hx[cc]
       
       # Compute the terrain derivatives       
       for rr in range(1,NZ):
              DZT[rr,:] = DDX_1D.dot(ZTL[rr,:] - z[rr])
              
       # Compute the coordinate surface at the edge of the Rayleigh layer
       ZRL = (1.0 - zRay / ZH) * hx + zRay
       
       return XL, ZTL, DZT, sigma, ZRL