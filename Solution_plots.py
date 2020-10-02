#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:21:22 2020

Production panel plot for solution fields

@author: TempestGuerra
"""

import shelve
import scipy.io as sio
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
'''
#%% SMOOTH STRATIFICATION
# Make the parent figure
fig = plt.figure(figsize=(12.0, 6.0))

la_afs = shelve.open('/home/jeg/scratch/restartDB_smooth025m', 'r')
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

# 2 X 2 subplot with all fields at the final time
for pp in range(4):
       plt.subplot(2,2,pp+1)
       
       if pp == 0:
              Q = np.reshape(SOL[udex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-0.25, vmax=0.25)
              plt.title('U (m/s)')
              plt.ylabel('Height (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
       elif pp == 1:
              Q = np.reshape(SOL[wdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-0.08, vmax=0.08)
              plt.title('W (m/s)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
              plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
       elif pp == 2:
              Q = np.reshape(SOL[pdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-4.5E-5, vmax=4.5E-5)
              plt.title('log-P (Pa)')
              plt.xlabel('Distance (km)')
              plt.ylabel('Height (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       elif pp == 3:
              Q = np.reshape(SOL[tdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-6.0E-4, vmax=6.0E-4)
              plt.title('log-Theta (K)')
              plt.xlabel('Distance (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
              plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
              
       fig.colorbar(ccheck, format='%.2E')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('python results/SolutionFields_smooth010m.png')
plt.show()
'''
#%% DISCRETE STRATIFICATION
# Make the parent figure
fig = plt.figure(figsize=(12.0, 6.0))

la_afs = shelve.open('/home/jeg/scratch/restartDB_discrete025m', 'r')
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

# 2 X 2 subplot with all fields at the final time
for pp in range(4):
       plt.subplot(2,2,pp+1)
       
       if pp == 0:
              dbound = 0.85
              Q = np.reshape(SOL[udex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-dbound, vmax=dbound)
              plt.title('U (m/s)')
              plt.ylabel('Height (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
       elif pp == 1:
              dbound = 0.2
              Q = np.reshape(SOL[wdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-dbound, vmax=dbound)
              plt.title('W (m/s)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
              plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
       elif pp == 2:
              dbound = 1.25E-4
              Q = np.reshape(SOL[pdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-dbound, vmax=dbound)
              plt.title('log-P (Pa)')
              plt.xlabel('Distance (km)')
              plt.ylabel('Height (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       elif pp == 3:
              dbound = 2.0E-3
              Q = np.reshape(SOL[tdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic, vmin=-dbound, vmax=dbound)
              plt.title('log-Theta (K)')
              plt.xlabel('Distance (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
              plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
       
       plt.clim(-dbound, dbound)
       norm = mpl.colors.Normalize(vmin=-dbound, vmax=dbound)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic), format='%.2E')
       #fig.colorbar(ccheck, format='%.2E')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('python results/SolutionFields_discrete025m.png')
plt.show()
'''
#%% SCHAR CASE WITH 50 M MOUNTAIN
# Make the parent figure
fig = plt.figure(figsize=(12.0, 6.0))

la_afs = shelve.open('/home/jeg/scratch/restartDB_exactBCSchar_025m', 'r')
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

# 2 X 2 subplot with all fields at the final time
for pp in range(4):
       plt.subplot(2,2,pp+1)
       
       if pp == 0:
              Q = np.reshape(SOL[udex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic)#, vmin=-0.55, vmax=0.55)
              plt.title('U (m/s)')
              plt.ylabel('Height (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
       elif pp == 1:
              Q = np.reshape(SOL[wdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic)#, vmin=-0.4, vmax=0.4)
              plt.title('W (m/s)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
              plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
       elif pp == 2:
              Q = np.reshape(SOL[pdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic)#, vmin=-6.25E-4, vmax=6.25E-4)
              plt.title('log-P (Pa)')
              plt.xlabel('Distance (km)')
              plt.ylabel('Height (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       elif pp == 3:
              Q = np.reshape(SOL[tdex,0], (NZ, NX+1), order='F')
              ccheck = plt.contourf(1.0E-3*X, 1.0E-3*Z, Q, 50, cmap=cm.seismic)#, vmin=-6.25E-4, vmax=6.25E-4)
              plt.title('log-Theta (K)')
              plt.xlabel('Distance (km)')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
              plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
              
       fig.colorbar(ccheck, format='%.2E')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('python results/SolutionFields_schar025m.png')
plt.show()
'''