#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:49:24 2019

@author: -
"""

import numpy as np

def computeAdjust4CBC(DIMS, numVar, varDex, latPeriodic, latInflow):
       # Get DIMS data
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # All DOF per variable
       rowsAll = set(np.array(range(0,OPS)))
       
       # Get prognostic ordering
       #iU = varDex[0]
       iW = varDex[1]
       iP = varDex[2]
       iT = varDex[3]
       
       # Compute BC index vectors for U and W (coupled top and bottom BC)
       
       # including corners
       #uldex = np.array(range(0, NZ))
       #urdex = np.array(range((OPS-NZ)-1, OPS))
       # excluding bottom corners
       uldex = np.array(range(1, NZ))
       urdex = np.array(range((OPS-NZ)+1, OPS))
       
       # including corners
       ubdex = np.array(range(0, OPS, NZ))
       utdex = np.array(range(NZ-1, OPS, NZ))
       # excluding corners
       #ubdex = np.array(range(1, OPS-2*NZ, NZ))
       #utdex = np.array(range(2*NZ-1, OPS-NZ, NZ))
              
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
       
       # Index all boundary DOF that can be diffused on
       latDex = np.unique(np.concatenate((uldex,urdex)))
       vrtDex = np.unique(np.concatenate((ubdex,utdex)))
       extDex = np.unique(np.concatenate((urdex,uldex,ubdex,utdex)))
       diffDex = (latDex, ubdex, utdex, extDex)
       
       # BC indices for static solution (per variable)
       if latPeriodic and not latInflow:
              # Periodic with no inflow condition
              rowsOutU = set()
              rowsOutW = set(utdex)
              rowsOutP = set()
              rowsOutT = set(utdex)
              # Indexing for static solver
              top = np.concatenate((wtdex, ttdex))
              rowsOutBC_static = set(top)
       elif not latPeriodic and latInflow:
              # Only inflow condition specified
              rowsOutU = set(uldex)
              rowsOutW = set(np.concatenate((uldex,utdex)))
              rowsOutP = set(uldex)
              rowsOutT = set(np.concatenate((uldex,utdex)))
              # Indexing for static solver
              left = np.concatenate((uldex, wldex, pldex, tldex))
              top = np.concatenate((wtdex, ttdex))
              rowsOutBC_static = set(np.concatenate((left, top)))
       elif latPeriodic and latInflow:
              '''
              # Periodic with inflow condition (pinned boundary)
              rowsOutU = set(np.concatenate((uldex,urdex)))
              rowsOutW = set(np.concatenate((uldex,urdex,utdex)))
              rowsOutP = set(np.concatenate((uldex,urdex)))
              rowsOutP = set(np.concatenate((uldex,urdex)))
              rowsOutT = set(np.concatenate((uldex,urdex,utdex)))
               # Indexing for static solver
              left = np.concatenate((uldex, wldex, pldex, tldex))
              right = np.concatenate((urdex, wrdex, prdex, trdex))
              top = np.concatenate((wtdex, ttdex))
              rowsOutBC_static = set(np.concatenate((left, right, top)))
              '''
              #'''
              # Periodic with inflow condition (pinned boundary)
              rowsOutU = set(np.concatenate((uldex,urdex)))
              rowsOutW = set(np.concatenate((uldex,urdex,utdex)))
              rowsOutP = set()
              rowsOutT = set(np.concatenate((uldex,urdex,utdex)))
               # Indexing for static solver
              left = np.concatenate((uldex, wldex, tldex))
              right = np.concatenate((urdex, wrdex, trdex))
              top = np.concatenate((wtdex, ttdex))
              rowsOutBC_static = set(np.concatenate((left, right, top)))
              #'''
       
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
       
       return uldex, urdex, ubdex, utdex, wbdex, ubcDex, wbcDex, pbcDex, tbcDex, rowsOutBC_transient, sysDex, diffDex