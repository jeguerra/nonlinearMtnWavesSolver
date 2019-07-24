#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:12:01 2019

@author: TempestGuerra
"""

import numpy as np
from HerfunChebNodesWeights import hefunclb
#from HerfunChebNodesWeights import hefuncm
from HerfunChebNodesWeights import cheblb
#from HerfunChebNodesWeights import chebpolym

def computeGrid(DIMS):
       # Get the domain dimensions
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       NX = DIMS[3]
       NZ = DIMS[4]
       
       # Compute the Hermite function and Chebyshev native grids
       alpha, whf = hefunclb(NX)
       xi, wcp = cheblb(NZ)
       
       # Compute the HF and Cheb matrices
       #HFM = hefuncm(NX, alpha, True)
       #CPM = chebpolym(NZ, xi)
       
       # Map reference 1D domains to physical 1D domains
       x = 0.5 * abs(L2 - L1) / np.amax(alpha) * alpha
       z = 0.5 * ZH * (1.0 - xi)
       
       # Return the REFS structure
       REFS = [x, z]
       
       return REFS