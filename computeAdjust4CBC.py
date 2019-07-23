#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:49:24 2019

@author: -
"""

import numpy as np

def computeAdjust4CBC(DIMS, numVar, varDex):
       # Get DIMS data
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Get prognostic ordering
       #iU = varDex[0]
       iW = varDex[1]
       #iP = varDex[2]
       #iT = varDex[3]
       
       # Compute BC index vectors for U and W (coupled top and bottom BC)
       ubdex = np.array(range(0, (OPS - NZ + 1), NZ))
       wbdex = np.add(ubdex, iW * OPS)
       utdex = np.array(range(NZ-1, OPS, NZ))
       wtdex = np.add(utdex, iW * OPS)
       
       # BC: w' = dh/dx (U + u') so that w' is set at top and bottom boundaries
       rowsOut = set(np.concatenate((wbdex, wtdex)))
       rowsAll = set(np.array(range(0,numVar*OPS)))
       
       # Compute set difference from all rows to rows to be taken out
       sysDex = rowsAll.difference(rowsOut)
       sysDex = sorted(sysDex)
       
       return ubdex, wbdex, sysDex