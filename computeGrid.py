#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:12:01 2019

@author: TempestGuerra
"""

import numpy as np
import HerfunChebNodesWeights as qxw
import matplotlib.pyplot as plt

def computeGrid(DIMS, Herm, Four, ChebCol, LegCol):
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
              zf = qxw.chebpolym(NZ-1, xi)
       elif LegCol:
              # Compute the Legendre native grids
              xi, wcp = qxw.leglb(NZ) #[-1 +1]
              z = 0.5 * ZH * (1.0 + xi)
              zf = qxw.legpolym(NZ-1, xi, False)
       else:
              z = np.linspace(0.0, ZH, num=NZ, endpoint=True)
       
       # Map reference 1D domains to physical 1D domains
       if Herm and not Four:
              alpha, whf = qxw.hefunclb(NX) #(-inf inf)
              x = 0.5 * abs(L2 - L1) / np.amax(alpha) * alpha
              xf = qxw.hefuncm(NX-1, alpha, False)
       elif Four and not Herm:
              x = np.linspace(L1, L2, num=NX+1, endpoint=True)
       else:
              
              if NX % 2 == 0:
                     NXH = int(0.5 * (NX + 2))
              else:
                     NXH = int(0.5 * (NX + 1))
              
              alpha, whf = qxw.cheblb(NXH)
              x1 = -0.5 * L1 * (alpha - 1.0)
              x2 = +0.5 * L2 * (1.0 + alpha)
              
              if NX % 2 == 0:
                     x = np.append(x1, x2[1:])
              else:
                     x = np.append(x1[0:-1], x2[1:])
                     
              #print(x)
       
       # Return the REFS structure
       REFS = [x, z]
       
       return REFS