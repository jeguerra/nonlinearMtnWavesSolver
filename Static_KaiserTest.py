#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:15:24 2020

STATIC KAISER MOUNTAIN TEST

@author: TempestGuerra
"""

# Set the solution type (MUTUALLY EXCLUSIVE)
StaticSolve = True
LinearSolve = False
NonLinSolve = False

# Set the grid type (NOT IMPLEMENTED)
HermCheb = True
UniformDelta = False

# Set 4th order compact finite difference derivatives switch
SparseDerivativesDynamics = False
SparseDerivativesDynSGS = False

# Set residual diffusion switch
ResDiff = False

# Set direct solution method (MUTUALLY EXCLUSIVE)
SolveFull = False
SolveSchur = True

# Set Newton solve initial and restarting parameters
toRestart = True # Saves resulting state to restart database
isRestart = False # Initializes from a restart database

# Set physical constants (dry air)
gc = 9.80601
P0 = 1.0E5
cp = 1004.5
Rd = 287.06
Kp = Rd / cp
cv = cp - Rd
gam = cp / cv
NBVP = 0.0125
PHYS = [gc, P0, cp, Rd, Kp, cv, gam, NBVP]

# Set grid dimensions and order
L2 = 1.0E4 * 3.0 * mt.pi
L1 = -L2
ZH = 31000.0
NX = 167 # FIX: THIS HAS TO BE AN ODD NUMBER!
NZ = 96
OPS = (NX + 1) * NZ
numVar = 4
NQ = OPS * numVar
iU = 0
iW = 1
iP = 2
iT = 3
varDex = [iU, iW, iP, iT]
DIMS = [L1, L2, ZH, NX, NZ, OPS]
# Make the equation index vectors for all DOF
udex = np.array(range(OPS))
wdex = np.add(udex, OPS)
pdex = np.add(wdex, OPS)
tdex = np.add(pdex, OPS)

# Background temperature profile
smooth3Layer = True
uniformStrat = False
T_in = [300.0, 228.5, 228.5, 248.5]
Z_in = [0.0, 1.1E4, 2.0E4, ZH]

# Background wind profile
uniformWind = False
JETOPS = [10.0, 16.822, 1.386]

# Set the Rayleigh options
depth = 6000.0
width = 16000.0
applyTop = True
applyLateral = True
mu = np.array([1.0E-2, 1.0E-2, 1.0E-2, 1.0E-2])
mu *= 1.0

# Set the terrain options
KAISER = 1 # Kaiser window profile
SCHAR = 2 # Schar mountain profile nominal (Schar, 2001)
EXPCOS = 3 # Even exponential and squared cosines product
EXPPOL = 4 # Even exponential and even polynomial product
INFILE = 5 # Data from a file (equally spaced points)
MtnType = KAISER
h0 = 100.0
aC = 5000.0
lC = 4000.0

if MtnType == KAISER:
       # When using this profile as the terrain
       kC = 10000.0
else:
       # When applying windowing to a different profile
       kC = L2 - width
       
HOPT = [h0, aC, lC, kC]

#% Transient solve parameters
DT = 0.05
HR = 5.0
rampTime = 900  # 10 minutes to ramp up U_bar
intMethodOrder = 3 # 3rd or 4th order time integrator
ET = HR * 60 * 60 # End time in seconds
OTI = 200 # Stride for diagnostic output
ITI = 1000 # Stride for image output
RTI = 1 # Stride for residual visc update