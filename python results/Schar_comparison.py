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
fig = plt.figure(figsize=(12.0, 4.0))

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
plt.xlabel('X (km)')
plt.ylabel('Z (km)')
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
plt.xlabel('X (km)')
plt.title('Approximate Free-Slip - W (m/s)')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

#%% Make common colorbar
axc = plt.subplot(1,3,3)
axc.set_axis_off()
fig.colorbar(ccheck, fraction=1.0)
plt.tight_layout()
plt.savefig('ScharTestComparisonW.png')


#%% Make plot for iterative solution to 250m Schar problem
fig = plt.figure(figsize=(12.0, 4.0))

la_afs = shelve.open('/home/jeg/scratch/ScharTest_Newton10_FreeSlipApprox_250m', 'r')
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
W = np.reshape(SOL[wdex,0], (NZ, NX+1), order='F')

plt.subplot(1,2,1)
ccheck = plt.contourf(1.0E-3 * X, 1.0E-3 * Z, W, 44, cmap=cm.seismic, vmin=-2.2, vmax=2.2)
#fig.colorbar(ccheck)
plt.xlim(-20.0, 20.0)
plt.ylim(0.0, 15.0)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
plt.xlabel('X (km)')
plt.ylabel('Z (km)')
plt.title('Estimated Steady Solution - W (m/s)')
plt.colorbar()
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

fname = '/media/jeg/FastDATA/linearMtnWavesSolver/python results/convergence250m_classical.txt'
from scipy.optimize import curve_fit
con_data = np.loadtxt(fname, delimiter=', ')

# Do an exponential curve fit to the total residual
def func(x, a, b):
       return -b * x + a

lp = 10
xdata = np.arange(0,lp)
ydata = np.log(con_data[0:lp,4])
popt, pcov = curve_fit(func, xdata, ydata, p0=[1.0E-3, 2.0], method='lm')
rate = popt[1]

# Make the nice paper plot
xdata = np.arange(0,con_data.shape[0])
fdata = func(xdata, *popt)

# Make the plots
plt.subplot(1,2,2)
plt.plot(xdata, con_data[:,4], 'kd-')
plt.plot(xdata, np.exp(fdata), 'r--')
plt.yscale('log')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.legend(('Total Residual', 'Convergence Rate = ' + '%.5f' % rate))
plt.xlabel('Newton Iteration')
plt.ylabel('L2-norm of Residual')
plt.title('Total Residual Convergence')
plt.tight_layout()
plt.savefig('ScharTestNewtonW.png')
