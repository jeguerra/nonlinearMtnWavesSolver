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
              zu = np.linspace(0.0, ZH, num=NZ)
              dz = abs(zu[1] - zu[0])
              
              zb = 0.5 * RLOPT[0]
              NI = int(NZ * (ZH - zb) / ZH)
              zi = np.linspace(0.0, ZH - zb, NI)
              dz2 = 2.0 * dz
              zt = np.linspace(ZH - zb + dz2, ZH, num=int(RLOPT[0] / dz2), endpoint=True)
              z = np.concatenate((zi, zt))
       
       # Map reference 1D domains to physical 1D domains
       if Herm and not Four:
              alpha, whf = qxw.hefunclb(NX) #(-inf inf)
              x = 0.5 * abs(L2 - L1) / np.amax(alpha) * alpha
       elif Four and not Herm:
              x = np.linspace(L1, L2, num=NX, endpoint=True)
       else:
              xu = np.linspace(L1, L2, num=NX, endpoint=True)
              dx = abs(xu[1] - xu[0])
              
              # Interior grid with 2X coarser in sponge layers
              xb = 0.5 * RLOPT[1]
              NI = int(NX * ((L2 - L1) - 2.0 * xb) / (L2 - L1))
              xi = np.linspace(L1 + xb, L2 - xb, NI)
              dx2 = 2.0 * dx
              xl = np.linspace(L1, L1 + xb, num=int(xb / dx2), endpoint=False)
              xr = np.linspace(L2 - xb + dx2, L2, num=int(xb / dx2), endpoint=True)
              x = np.concatenate((xl, xi, xr))
       
       # Return the REFS structure
       REFS = [x, z]
       
       return REFS