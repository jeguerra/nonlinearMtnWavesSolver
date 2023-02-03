#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:23:58 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
#import matplotlib.pyplot as plt

def computeTerrainDecayFunctions(xi, ang, StaticSolve):
       
       if StaticSolve:
              # Nominal Hybrid coordinate
              m = 0.2
              mi = 1.0 / m
              dzdh = np.sinh(mi * (1.0 - xi)) / np.sinh(mi)
              d_dzdh_dxi = -mi * np.cosh(mi * (1.0 - xi)) / np.sinh(mi)
       else:
              # First pass [A=0.3, p=20, m=0.2]
              # Second pass [A=0.4, p=10, m=0.25]
              # Third pass [A=0.25, p=25, m=0.25]
              A = 0.25
              p = 25
              
              # Guellrich improvement to hybrid coordinate
              cosvar = np.power(np.cos(A * ang), p)
              tanvard = A * mt.pi * np.tan(A * ang)
              
              # Guellrich Hybrid coordinate
              m = 0.25
              mi = 1.0 / m
              hybrid = np.sinh(mi * (1.0 - xi)) / np.sinh(mi)
              dhybrid = -mi * np.cosh(mi * (1.0 - xi)) / np.sinh(mi)
              dzdh = cosvar * np.sinh(mi * (1.0 - xi)) / np.sinh(mi)
              d_dzdh_dxi = cosvar * (dhybrid - p * hybrid * tanvard) 
       
       return dzdh, d_dzdh_dxi

def computeGuellrichDomain2D(DIMS, x, z, zRay, hx, dhdx, StaticSolve):
       # Get data from DIMS and REFS
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       
       # Compute the flat XZ mesh (computational domain)
       HTZL, dummy = np.meshgrid(hx, z)
       XL, ZL = np.meshgrid(x,z)
       
       # Make the global array of terrain height and slope features
       ZTL = np.zeros((NZ,NX))
       DZT = np.zeros((NZ,NX))
       
       # High Order Improved Guellrich coordinate 3 parameter function
       xi = 1.0 / ZH * ZL
       ang = 1.0 / 3.0 * mt.pi * xi
       dzdh, d_dzdh_dxi = computeTerrainDecayFunctions(xi, ang, StaticSolve)
       
       for rr in range(NZ):
              ZTL[rr,:] = (dzdh[rr,0] * hx) + ZL[rr,:]
              DZT[rr,:] = dzdh[rr,0] * dhdx
       
       dxidz = (1.0 + (HTZL / ZH * d_dzdh_dxi))
       sigma = np.reciprocal(dxidz)
              
       #plt.plot(z, dzdh[:,0])
       
       # Compute the coordinate surface at edge of Rayleigh layer
       xi = 1.0 / ZH * zRay
       ang = 1.0 / 3.0 * mt.pi * xi
       dzdh, d_dzdh_dxi = computeTerrainDecayFunctions(xi, ang, StaticSolve)
       
       ZRL = (dzdh * hx) + zRay
       
       # Compute the local grid lengths at each node
       DXM = np.gradient(XL, np.arange(XL.shape[1]), axis=1, edge_order=1)
       DZM = np.gradient(ZTL, np.arange(ZTL.shape[0]), axis=0, edge_order=1)
       
       return XL, ZTL, DZT, sigma, ZRL, DXM, DZM

def computeStretchedDomain2D(DIMS, REFS, zRay, hx, dhdx):
       # Get data from DIMS and REFS
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       
       # Get REFS data
       x = REFS[0]
       z = REFS[1]
       DDX_1D = REFS[2]
       
       # Compute the flat XZ mesh
       DZT, dummy = np.meshgrid(dhdx,z);
       XL, ZL = np.meshgrid(x,z);
       
       # Make the global array of terrain height and slope features
       ZTL = np.zeros((NZ,NX))
       
       sigma = np.ones((NZ,NX))
       for cc in range(NX):
              thisZH = ZH - hx[cc]
              sigma[:,cc] *= (ZH / thisZH)
              ZTL[:,cc] = ZL[:,cc] * thisZH / ZH
              ZTL[:,cc] += (ZH - ZTL[-1,cc])
       
       # Compute the terrain derivatives       
       for rr in range(1,NZ):
              DZT[rr,:] = DDX_1D.dot(ZTL[rr,:] - z[rr])
              
       # Compute the coordinate surface at the edge of the Rayleigh layer
       ZRL = (1.0 - zRay / ZH) * hx + zRay
       
       # Compute the local grid lengths at each node
       DXM = np.gradient(XL, np.arange(XL.shape[1]), axis=1, edge_order=1)
       DZM = np.gradient(ZTL, np.arange(ZTL.shape[0]), axis=0, edge_order=1)
       
       return XL, ZTL, DZT, sigma, ZRL, DXM, DZM