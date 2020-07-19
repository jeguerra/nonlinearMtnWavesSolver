#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:51:46 2020

Production panel plot for Schar test case comparison
2X2 panel

@author: TempestGuerra
"""

import shelve
import scipy.io as sio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# Make the parent figure
fig = plt.figure(figsize=(12.0, 6.0))

#%% Load in the classical linear solution from Matlab (approximate free-slip)
clin = sio.loadmat('../matlab backup/AnalyticalSchar.mat', squeeze_me=True)
XM = clin['X']
ZM = clin['Z']
WM = clin['w']

plt.subplot(1,3,1)
ccheck = plt.contourf(1.0E-3 * XM, 1.0E-3 * ZM, WM.T, 44, cmap=cm.seismic, vmin=-2.2, vmax=2.2)
#fig.colorbar(ccheck)
plt.xlim(-20.0, 20.0)
plt.ylim(0.0, 15.0)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
plt.xlabel('Distance (km)')
plt.title('Classical Fourier - W (m/s)')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

#%% Load in the limited area reproduction with approximate free-slip
la_afs = shelve.open('/home/jeg/scratch/ScharTest_Classical_FreeSlipApprox_250m', 'r')
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
x1 = X[0,:]
Z = REFS[5]
W1 = np.reshape(SOL[wdex,0], (NZ, NX+1), order='F')

plt.subplot(1,3,2)
ccheck = plt.contourf(1.0E-3 * X, 1.0E-3 * Z, W1, 44, cmap=cm.seismic, vmin=-2.2, vmax=2.2)
#fig.colorbar(ccheck)
plt.xlim(-20.0, 20.0)
plt.ylim(0.0, 15.0)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
plt.xlabel('Distance (km)')
plt.title('Approximate Free-Slip - W (m/s)')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

#%% Make common colorbar
axc = plt.subplot(1,3,3)
axc.set_axis_off()
fig.colorbar(ccheck, fraction=1.0)
plt.tight_layout()
plt.savefig('ScharTestComparisonW.png')

# Make the parent figure
fig = plt.figure(figsize=(12.0, 6.0))

#%% Load in the limited area reproduction with approximate free-slip
la_afs = shelve.open('/home/jeg/scratch/ScharTest_Classical_FreeSlipApprox_250m', 'r')
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
x2 = X[0,:]
Z = REFS[5]
U = np.reshape(SOL[udex,0], (NZ, NX+1), order='F')

plt.subplot(1,3,1)
ccheck = plt.contourf(1.0E-3 * X, 1.0E-3 * Z, U, 44, cmap=cm.seismic, vmin=-2.2, vmax=2.2)
#fig.colorbar(ccheck)
plt.xlim(-20.0, 20.0)
plt.ylim(0.0, 15.0)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
plt.xlabel('Distance (km)')
plt.title('Approximate Free-Slip - U (m/s)')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

#%% Load in the limited area reproduction with approximate free-slip
la_afs = shelve.open('/home/jeg/scratch/ScharTest_Classical_FreeSlipExact_250m', 'r')
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
U = np.reshape(SOL[udex,0], (NZ, NX+1), order='F')
W2 = np.reshape(SOL[wdex,0], (NZ, NX+1), order='F')

plt.subplot(1,3,2)
ccheck = plt.contourf(1.0E-3 * X, 1.0E-3 * Z, U, 44, cmap=cm.seismic, vmin=-2.2, vmax=2.2)
#fig.colorbar(ccheck)
plt.xlim(-20.0, 20.0)
plt.ylim(0.0, 15.0)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
plt.xlabel('Distance (km)')
plt.title('Exact Free-Slip - U (m/s)')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

#%% Make common colorbar
axc = plt.subplot(1,3,3)
axc.set_axis_off()
fig.colorbar(ccheck, fraction=1.0)
plt.tight_layout()
plt.savefig('FreeSlipComparisonU.png')

#%% Compare lower boundary condition
fig = plt.figure(figsize=(12.0, 6.0))
plt.subplot(1,2,1)
plt.plot(1.0E-3 * XM[0,:], WM.T[0,:], 'k')
plt.plot(1.0E-3 * x1, W1[0,:], 'b--')
plt.plot(1.0E-3 * x2, W2[0,:], 'g--')
plt.legend(['Classical Solution', 'Approximate Free-Slip','Exact Free-Slip'])
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.title('Boundary W (m/s) - Comparison')
plt.xlim(-10.0, 10.0)
'''
plt.subplot(3,1,2)
plt.plot(1.0E-3 * x2, W2[0,:], 'k')
plt.plot(1.0E-3 * x1, W1[0,:], 'b--')
plt.legend(['Exact Free-Slip', 'Approximate Free-Slip'])
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.title('Boundary W (m/s) - Exact vs. Approximate')
plt.xlim(-20.0, 20.0)
'''
plt.subplot(1,2,2)
plt.plot(1.0E-3 * x1, W2[0,:] - W1[0,:], 'k')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.title('Boundary W (m/s) - Difference')
plt.xlim(-10.0, 10.0)
plt.xlabel('Distance (km)')

plt.tight_layout()
plt.savefig('ScharTestComparison_WBC.png')


