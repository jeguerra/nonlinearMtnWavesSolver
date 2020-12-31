#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:49:24 2019

@author: -
"""

import numpy as np

def computeAdjust4CBC(DIMS, numVar, varDex, bcType):
       # Get DIMS data
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = (NX+1) * NZ
       
       # All DOF per variable
       rowsAll = set(np.array(range(0,OPS)))
       
       # Get prognostic ordering
       #iU = varDex[0]
       iW = varDex[1]
       iP = varDex[2]
       iT = varDex[3]
       
       # Compute BC index vectors for U and W (coupled top and bottom BC)
       # including corners
       ubdex = np.array(range(0, OPS, NZ))
       utdex = np.array(range(NZ-1, OPS, NZ))
       
       # including corners
       uldex1 = np.array(range(ubdex[0], NZ))
       urdex1 = np.array(range(ubdex[-1], OPS))
       # excluding corners at terrain boundary
       uldex2 = np.array(range(ubdex[0]+1, NZ))
       urdex2 = np.array(range(ubdex[-1]+1, OPS))
              
       wldex = np.add(uldex2, iW * OPS)
       wrdex = np.add(urdex2, iW * OPS)
       wbdex = np.add(ubdex, iW * OPS)
       wtdex = np.add(utdex, iW * OPS)
       
       pldex = np.add(uldex2, iP * OPS)
       prdex = np.add(urdex2, iP * OPS)
       #pbdex = np.add(ubdex, iP * OPS)
       #ptdex = np.add(utdex, iP * OPS)
       
       tldex = np.add(uldex2, iT * OPS)
       trdex = np.add(urdex2, iT * OPS)
       #tbdex = np.add(ubdex, iT * OPS)
       #ttdex = np.add(utdex, iT * OPS)
       
       # Index all boundary DOF that can be diffused on
       latDex = np.unique(np.concatenate((uldex2,urdex2)))
       #vrtDex = np.unique(np.concatenate((ubdex,utdex)))
       extDex = np.unique(np.concatenate((urdex2,uldex2,ubdex,utdex)))
       diffDex = (latDex, ubdex, utdex, extDex)
       
       # BC indices for static solution (per variable)
       if bcType == 1:
              # Inflow condition on kinematic variables
              rowsOutU = set(uldex2)
              rowsOutW = set(np.concatenate((uldex2,utdex)))
              rowsOutP = set()
              rowsOutT = set()
              # Indexing for static solver
              left = np.concatenate((uldex2, wldex))
              top = wtdex
              rowsOutBC_static = set(np.concatenate((left, top)))
       elif bcType == 2:
              # Pinned condition on kinematic variables
              rowsOutU = set(np.concatenate((uldex2,urdex1)))
              rowsOutW = set(np.concatenate((uldex2,urdex1,utdex)))
              rowsOutP = set()
              rowsOutT = set()
               # Indexing for static solver
              left = np.concatenate((uldex2, wldex))
              right = np.concatenate((urdex2, wrdex))
              top = wtdex
              rowsOutBC_static = set(np.concatenate((left, right, top)))
       elif bcType == 3:
              # Inflow condition on UWPT
              rowsOutU = set(uldex2)
              rowsOutW = set(np.concatenate((uldex2,utdex)))
              rowsOutP = set(uldex2)
              rowsOutT = set(uldex2)
              # Indexing for static solver
              left = np.concatenate((uldex2, wldex, pldex, tldex))
              top = wtdex
              rowsOutBC_static = set(np.concatenate((left, top)))
       elif bcType == 4:
              # Pinned condition on UWPT
              rowsOutU = set(np.concatenate((uldex2,urdex2)))
              rowsOutW = set(np.concatenate((uldex2,urdex2,utdex)))
              rowsOutP = set(np.concatenate((uldex2,urdex2)))
              rowsOutT = set(np.concatenate((uldex2,urdex2)))
               # Indexing for static solver
              left = np.concatenate((uldex2, wldex, pldex, tldex))
              right = np.concatenate((urdex2, wrdex, prdex, trdex))
              top = wtdex
              rowsOutBC_static = set(np.concatenate((left, right, top)))
       elif bcType == 5:
              # Pinned condition on UWT
              rowsOutU = set(np.concatenate((uldex2,urdex2)))
              rowsOutW = set(np.concatenate((uldex2,urdex2,utdex)))
              rowsOutP = set()
              rowsOutT = set(np.concatenate((uldex2,urdex2)))
               # Indexing for static solver
              left = np.concatenate((uldex2, wldex, tldex))
              right = np.concatenate((urdex2, wrdex, trdex))
              top = wtdex
              rowsOutBC_static = set(np.concatenate((left, right, top)))
       
       # Indexing arrays for static solution
       ubcDex = rowsAll.difference(rowsOutU); ubcDex = sorted(ubcDex)
       wbcDex = rowsAll.difference(rowsOutW); wbcDex = sorted(wbcDex)
       pbcDex = rowsAll.difference(rowsOutP); pbcDex = sorted(pbcDex)
       tbcDex = rowsAll.difference(rowsOutT); tbcDex = sorted(tbcDex)
       
       # W is treated as an essential BC at terrain in solution by direct substitution
       rowsOutBC_transient = (sorted(rowsOutU), sorted(rowsOutW), \
                              sorted(rowsOutP), sorted(rowsOutT))
              
       # All DOF for all variables
       rowsAll = set(np.array(range(0,numVar*OPS)))
       # Compute set difference from all rows to rows to be taken out LINEAR
       sysDex = rowsAll.difference(rowsOutBC_static)
       sysDex = sorted(sysDex)
       
       return uldex1, urdex1, ubdex, utdex, wbdex, ubcDex, wbcDex, pbcDex, tbcDex, rowsOutBC_transient, sysDex, diffDex