#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:51:46 2020

Production panel plot for Schar test case comparison
2X2 panel

@author: TempestGuerra
"""

import dbm.gnu
import shelve
import scipy.io as sio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# Make the parent figure
fig = plt.figure(figsize=(12.0, 6.0))

# Load in the classical linear solution from Matlab (approximate free-slip)
clin = sio.loadmat('../matlab backup/AnalyticalSchar.mat', squeeze_me=True)
XM = clin['X']
ZM = clin['Z']
WM = clin['w']

plt.subplot(2,2,1)
ccheck = plt.contourf(1.0E-3 * XM, 1.0E-3 * ZM, WM.T, 51, cmap=cm.seismic, vmin=-2.0, vmax=2.0)
fig.colorbar(ccheck)
plt.xlim(-30.0, 30.0)
plt.ylim(0.0, 15.0)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
plt.title('Classical Fourier Solution - W (m/s)')

# Load in the limited area reproduction with approximate free-slip
#la_afs = shelve.Shelf(gdbm.open('paper_results/ScharTest_Classical_NoSlipApprox_250m', 'r'))
la_afs = shelve.open('paper_results/ScharTest_Classical_NoSlipApprox_250m', 'r')
DIMS = la_afs['DIMS']
REFS = la_afs['REFS']
SOL = la_afs['SOLT']

# Make the equation index vectors for all DOF
numVar = 4
NX = DIMS[3]
NZ = DIMS[4]
OPS = DIMS[5]
udex = np.array(range(OPS))
wdex = np.add(udex, OPS)
pdex = np.add(wdex, OPS)
tdex = np.add(pdex, OPS)
       
X = REFS[4]
Z = REFS[5]
W = np.reshape(SOL[wdex,0], (NZ, NX+1), order='F')

plt.subplot(2,2,2)
ccheck = plt.contourf(1.0E-3 * X, 1.0E-3 * Z, WM, 51, cmap=cm.seismic, vmin=-2.0, vmax=2.0)
fig.colorbar(ccheck)
plt.xlim(-30.0, 30.0)
plt.ylim(0.0, 15.0)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
plt.title('Approximate Free-Slip Hermite/Chebyshev Solution - W (m/s)')

