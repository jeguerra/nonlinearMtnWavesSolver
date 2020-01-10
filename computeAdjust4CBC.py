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
       uldex = np.array(range(0, NZ)) # include the corner node
       urdex = np.array(range(OPS-NZ, OPS))
       ubdex = np.array(range(0, (OPS - NZ + 1), NZ))
       utdex = np.array(range(NZ-1, OPS, NZ))
       
       wldex = np.add(uldex, iW * OPS)
       wrdex = np.add(urdex, iW * OPS)
       wbdex = np.add(ubdex, iW * OPS)
       wtdex = np.add(utdex, iW * OPS)
       
       pldex = np.add(uldex, iP * OPS)
       prdex = np.add(urdex, iP * OPS)
       pbdex = np.add(ubdex, iP * OPS)
       ptdex = np.add(utdex, iP * OPS)
       
       tldex = np.add(uldex, iT * OPS)
       trdex = np.add(urdex, iT * OPS)
       tbdex = np.add(ubdex, iT * OPS)
       ttdex = np.add(utdex, iT * OPS)
       
       # Index all boundary DOF
       extDex = set(np.concatenate((uldex,urdex,ubdex,utdex, \
                                    wldex,wrdex,wbdex,wtdex, \
                                    pldex,prdex,pbdex,ptdex, \
                                    tldex,trdex,tbdex,ttdex)))
       extDex = sorted(extDex)
       
       # Local block-wide indices
       uldex = np.array(range(1, NZ)) # exclude the corner node
       wldex = np.add(uldex, iW * OPS)
       pldex = np.add(uldex, iP * OPS)
       tldex = np.add(uldex, iT * OPS)
       
       rowsOutU = set(uldex)
       rowsOutW = set(np.unique(np.concatenate((uldex,utdex))))
       rowsOutP = set(uldex)
       rowsOutT = set(np.unique(np.concatenate((uldex,utdex))))
       rowsAll = set(np.array(range(0,OPS)))
       
       ubcDex = rowsAll.difference(rowsOutU); ubcDex = sorted(ubcDex)
       wbcDex = rowsAll.difference(rowsOutW); wbcDex = sorted(wbcDex)
       pbcDex = rowsAll.difference(rowsOutP); pbcDex = sorted(pbcDex)
       tbcDex = rowsAll.difference(rowsOutT); tbcDex = sorted(tbcDex)
       
       # U and W at terrain boundary are NOT treated as essential BC in solution by Lagrange Multipliers
       rowsOutBC_static = set(np.unique(np.concatenate((uldex, wldex, pldex, tldex, wtdex, ttdex))))
       # W is treated as an essential BC at terrain in solution by direct substitution
       rowsOutBC_transient = set(np.unique(np.concatenate((uldex, wldex, pldex, tldex, wbdex, wtdex, ttdex))))
       # All DOF
       rowsAll = set(np.array(range(0,numVar*OPS)))
       
       # Compute set difference from all rows to rows to be taken out LINEAR
       sysDex = rowsAll.difference(rowsOutBC_static)
       sysDex = sorted(sysDex)
       # DOF's not dynamically updated in static solution (use Lagrange Multiplier)
       zeroDex_stat = sorted(rowsOutBC_static)
       # DOF's not dynamically updated in transient solution (use direct BC substitution)
       zeroDex_tran = sorted(rowsOutBC_transient)
       
       return ubdex, utdex, wbdex, pbdex, tbdex, ubcDex, wbcDex, pbcDex, tbcDex, zeroDex_stat, zeroDex_tran, sysDex, extDex