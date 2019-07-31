#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:47:32 2019

@author: -
"""

import numpy as np
#import matplotlib.pyplot as plt
import HerfunChebNodesWeights as hcnw

def computeHorizontalInterp(DIMS, NXI, FLD, HF_TRANS):
       # Get DIMS data
       NX = DIMS[3]
       
       # Check
       if NXI <= 0:
              print('ERROR: Invalid number of points in new grid! ', NXI)
              return FLD
       
       # Compute the new horizontal reference grid (linear space)
       xh, dummy = hcnw.hefunclb(NX)
       xmax = np.amax(xh)
       xmin = np.amin(xh)
       xi = np.linspace(xmin, xmax, num=NXI, endpoint=True)
       
       # Compute coefficients for the height field
       #hcoeffs = np.matmul(HF_TRANS, ZTL.T)
       # Compute coefficients for the variable field
       fcoeffs = np.matmul(HF_TRANS, FLD.T)
       
       # Compute backward transform to new grid
       HFM = hcnw.hefuncm(NX-1, xi, True)
       #HFM_native = hcnw.hefuncm(NX-1, xh, True)
       
       #plt.figure()
       #plt.plot(xh, HFM_native[0,:])
       #plt.figure()
       #plt.plot(xi, HFM[0,:])
       
       # Apply the backward transforms
       #ZTLI = np.matmul(HFM.T, hcoeffs)
       FLDI = np.matmul(HFM.T, fcoeffs)
       
       # Make a new XLI meshgrid
       #varray = np.array(range(NZI))
       #XLI, dummy = np.meshgrid(L2 / xmax * xi, varray)
       
       return FLDI.T