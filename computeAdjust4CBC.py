#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:49:24 2019

@author: -
"""

import numpy as np

def computeAdjust4CBC(DIMS, numVar, varDex, bcType):
       # Get DIMS data
       #NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
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
       
       # Index all boundary DOF that can be diffused on
       diffDex = (uldex1, urdex1, ubdex, utdex)
       
       isInflow = False
       
       # BC indices for static solution (per variable)
       if bcType == 1:
              if isInflow:
                     # Inflow condition on UWPT STATIC SOLUTION
                     rowsOutU = set(uldex2)
                     rowsOutW = set(np.concatenate((uldex2,utdex)))
                     rowsOutP = set(uldex2)
                     rowsOutT = set(uldex2)
                     # Indexing for static solver
                     left = np.concatenate((uldex2, uldex2 + iW*OPS, uldex2 + iP*OPS, uldex2 + iT*OPS))
                     top = utdex + iW*OPS
                     rowsOutBC_static = set(np.concatenate((left, top)))
              else:
                     # Inflow condition on UWPT TRANSIENT SOLUTION
                     rowsOutU = set(np.concatenate((uldex2,urdex2)))
                     rowsOutW = set(np.concatenate((uldex2,urdex2,utdex)))
                     rowsOutP = set(np.concatenate((uldex2,urdex2)))
                     #rowsOutP = set(uldex2)
                     rowsOutT = set(np.concatenate((uldex2,urdex2)))
              
                      # Indexing for static solver
                     left = np.concatenate((uldex2, uldex2 + iW*OPS, uldex2 + iP*OPS, uldex2 + iT*OPS))
                     right = np.concatenate((urdex2, urdex2 + iW*OPS, urdex2 + iP*OPS, urdex2 + iT*OPS))
                     #left = np.concatenate((uldex2, uldex2 + iW*OPS, uldex2 + iP*OPS, uldex2 + iT*OPS))
                     #right = np.concatenate((urdex2, urdex2 + iW*OPS, urdex2 + iT*OPS))
                     top = utdex + iW*OPS
                     rowsOutBC_static = set(np.concatenate((left, right, top)))
       elif bcType == 2:
              if isInflow:
                     # Inflow condition on UWPT STATIC SOLUTION
                     rowsOutU = set(uldex1)
                     rowsOutW = set(np.concatenate((uldex1,utdex)))
                     rowsOutP = set(uldex1)
                     rowsOutT = set(uldex1)
                     # Indexing for static solver
                     left = np.concatenate((uldex1, uldex1 + iW*OPS, uldex1 + iP*OPS, uldex1 + iT*OPS))
                     top = utdex + iW*OPS
                     rowsOutBC_static = set(np.concatenate((left, top)))
              else:
                     # Inflow condition on UWPT TRANSIENT SOLUTION
                     rowsOutU = set(np.concatenate((uldex1,urdex1)))
                     rowsOutW = set(np.concatenate((uldex1,urdex1,utdex)))
                     #rowsOutP = set(uldex1)
                     rowsOutP = set(np.concatenate((uldex1,urdex1)))
                     rowsOutT = set(np.concatenate((uldex1,urdex1)))
              
                      # Indexing for static solver
                     left = np.concatenate((uldex1, uldex1 + iW*OPS, uldex1 + iP*OPS, uldex1 + iT*OPS))
                     right = np.concatenate((urdex1, urdex1 + iW*OPS, urdex1 + iP*OPS, urdex1 + iT*OPS))
                     #left = np.concatenate((uldex1, uldex1 + iW*OPS, uldex1 + iP*OPS, uldex1 + iT*OPS))
                     #right = np.concatenate((urdex1, urdex1 + iW*OPS, urdex1 + iT*OPS))
                     top = utdex + iW*OPS
                     rowsOutBC_static = set(np.concatenate((left, right, top)))
              #'''
              '''
              # Kinematic essential BC
              rowsOutU = set(np.concatenate((uldex1,urdex1)))
              rowsOutW = set(np.concatenate((uldex1,urdex1,utdex)))
              rowsOutP = set()
              rowsOutT = set()
       
               # Indexing for static solver
              left = np.concatenate((uldex1, uldex1 + iW*OPS))
              right = np.concatenate((urdex1, urdex1 + iW*OPS))
              top = utdex + iW*OPS
              rowsOutBC_static = set(np.concatenate((left, right, top)))
              '''
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
       
       return uldex1, urdex1, ubdex, utdex, ubdex + iW*OPS, ubcDex, wbcDex, pbcDex, tbcDex, rowsOutBC_transient, sysDex, diffDex