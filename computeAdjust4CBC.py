#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:49:24 2019

@author: -
"""

import numpy as np

def computeAdjust4CBC(DIMS, numVar, varDex):
       # Get DIMS data
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Get prognostic ordering
       #iU = varDex[0]
       iW = varDex[1]
       iP = varDex[2]
       iT = varDex[3]
       
       # Compute BC index vectors for U and W (coupled top and bottom BC)
       ubdex = np.array(range(0, (OPS - NZ + 1), NZ))
       utdex = np.array(range(NZ-1, OPS, NZ))
       wbdex = np.add(ubdex, iW * OPS)
       wtdex = np.add(utdex, iW * OPS)
       pbdex = np.add(ubdex, iP * OPS)
       ptdex = np.add(utdex, iP * OPS)
       tbdex = np.add(ubdex, iT * OPS)
       ttdex = np.add(utdex, iT * OPS)
       
       # BC: w' = dh/dx (U + u') so that w' is at top and bottom boundaries
       rowsOut = set(np.concatenate((wbdex, wtdex)))
       rowsTopBot = set(np.concatenate((wbdex, wtdex, pbdex, ptdex, tbdex, ttdex)))
       rowsAll = set(np.array(range(0,numVar*OPS)))
       
       # Compute set difference from all rows to rows to be taken out
       sysDex = rowsAll.difference(rowsOut)
       sysDex = sorted(sysDex)
       
       # Compute set difference for all DOF interior to vertical boundaries
       intDex = rowsAll.difference(rowsTopBot)
       intDex = sorted(intDex)
       
       return ubdex, wbdex, sysDex, intDex