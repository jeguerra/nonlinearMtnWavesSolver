#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:12:01 2019

@author: TempestGuerra
"""

import math as mt
import numpy as np
import HerfunChebNodesWeights as qxw

def computeGrid(DIMS, RLOPT, Herm, Four, ChebCol, LegCol):
       # Get the domain dimensions
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       NX = DIMS[3]
       NZ = DIMS[4]
       
       if ChebCol:
              # Compute the Chebyshev native grids
              xi, wcp = qxw.cheblb(NZ) #[-1 +1]
              z = 0.5 * ZH * (1.0 + xi)
       elif LegCol:
              # Compute the Legendre native grids
              xi, wcp = qxw.leglb(NZ) #[-1 +1]
              z = 0.5 * ZH * (1.0 + xi)
       else:
              z = np.linspace(0.0, ZH, num=NZ)
       
       # Map reference 1D domains to physical 1D domains
       if Herm and not Four:
              alpha, whf = qxw.hefunclb(NX) #(-inf inf)
              x = 0.5 * abs(L2 - L1) / np.amax(alpha) * alpha
       elif Four and not Herm:
              x = np.linspace(L1, L2, num=NX, endpoint=True)
       else:
              x = np.linspace(L1, L2, num=NX, endpoint=True)
       
       # Return the REFS structure
       REFS = [x, z]
       
       return REFS