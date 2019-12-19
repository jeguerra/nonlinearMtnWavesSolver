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
       uldex = np.array(range(1, NZ)) # exclude the corner node
       #urdex = np.array(range(OPS-NZ, OPS))
       ubdex = np.array(range(0, (OPS - NZ + 1), NZ))
       utdex = np.array(range(NZ-1, OPS, NZ))
       wbdex = np.add(ubdex, iW * OPS)
       wtdex = np.add(utdex, iW * OPS)
       pbdex = np.add(ubdex, iP * OPS)
       #ptdex = np.add(utdex, iP * OPS)
       tbdex = np.add(ubdex, iT * OPS)
       ttdex = np.add(utdex, iT * OPS)
       
       # Local block-wide indices
       rowsOutU = set(uldex)
       rowsOutW = set(utdex)
       rowsOutT = set(utdex)
       rowsAll = set(np.array(range(0,OPS)))
       
       ubcDex = rowsAll.difference(rowsOutU); ubcDex = sorted(ubcDex)
       wbcDex = rowsAll.difference(rowsOutW); wbcDex = sorted(wbcDex)
       tbcDex = rowsAll.difference(rowsOutT); tbcDex = sorted(tbcDex)
       
       # BC: W' and Theta' = 0.0 at the top boundary, U' = 0 at the left boundary
       rowsOutBC_static = set(np.concatenate((uldex, wtdex, ttdex)))
       rowsOutBC_transient = set(np.concatenate((uldex, wbdex, wtdex, ttdex)))
       # All DOF
       rowsAll = set(np.array(range(0,numVar*OPS)))
       
       # Compute set difference from all rows to rows to be taken out LINEAR
       sysDex = rowsAll.difference(rowsOutBC_static)
       sysDex = sorted(sysDex)
       zeroDex = sorted(rowsOutBC_transient)
       
       return ubdex, utdex, wbdex, pbdex, tbdex, ubcDex, wbcDex, tbcDex, zeroDex, sysDex