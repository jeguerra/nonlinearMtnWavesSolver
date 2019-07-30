#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:47:32 2019

@author: -
"""

import numpy as np
#import scipy.interpolate as spint
import HerfunChebNodesWeights as hcnw

def computeHorizontalInterp(DIMS, NXI, NZI, ZTL, FLD, HF_TRANS):
       # Get DIMS data
       L2 = DIMS[1]
       NX = DIMS[3]
       
       # Check
       if NXI <= 0:
              print('ERROR: Invalid number of points in new grid! ', NZI)
              return FLD
       
       # Compute the new horizontal reference grid (linear space)
       xh, dummy = hcnw.hefunclb(NXI)
       xmax = np.amax(xh)
       xmin = np.amin(xh)
       xi = np.linspace(xmin, xmax, num=NXI, endpoint=True)
       
       # Compute coefficients for the height field
       hcoeffs = HF_TRANS.dot(ZTL.T)
       # Compute coefficients for the variable field
       fcoeffs = HF_TRANS.dot(FLD.T)
       
       # Compute backward transform to new grid
       HFM = hcnw.hefuncm(NX-1, xi, True)
       
       # Apply the backward transforms
       ZTLI = (HFM.T).dot(hcoeffs)
       FLDI = (HFM.T).dot(fcoeffs)
       
       # Make a new XLI meshgrid
       varray = np.array(range(NZI))
       XLI, dummy = np.meshgrid(L2 / xmax * xi, varray)
       
       return FLDI.T, XLI, ZTLI.T