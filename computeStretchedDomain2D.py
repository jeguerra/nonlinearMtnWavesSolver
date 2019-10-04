#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:23:58 2019

@author: TempestGuerra
"""

import numpy as np

def computeStretchedDomain2D(DIMS, REFS, hx, dhdx):
       # Get data from DIMS and REFS
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       
       # input REFS = [x, z, HFM, whf, CPM, wcp]
       x = REFS[0]
       z = REFS[1]
       
       # Compute the flat XZ mesh
       HTZL, dummy = np.meshgrid(hx,z);
       XL, ZL = np.meshgrid(x,z);
       
       # Make the global array of terrain height and slope features
       ZTL = np.zeros((NZ,NX))
       
       sigma = []
       for cc in range(NX):
              thisZH = ZH - hx[cc]
              sigma.append(ZH / thisZH)
              ZTL[:,cc] *= thisZH / ZH
              ZTL[:,cc] += hx[cc]
       
       return XL, ZTL, sigma