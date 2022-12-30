#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:35:16 2020

@author: TempestGuerra
"""
import math as mt

class TestCase:
       
       '''
       setUserData ARGS:
              NX: Odd number specifying horizontal grid size
              NZ: Integer specifying vertical grid size
              XF: Factor in kilometers for X domain length
              ZF: Factor in kilometers for Z domain length
              T0: Temperature at Z=0
              depth: Sponge layer depth (vertical)
              width: Sponge layer width (horizontal)
              rlf: Sponge layer strength factor
              h0: Maximum height for terrain profile
              NBVF: Brunt-Vaisala Frequency at Z=0
              Mountain: Profile type KAISER = 1, SCHAR = 2
              kC: Controls the width of KAISER profile in meters
              latBC: type of lateral boundary condition to use
       '''
       
       def __init__(self, TestName):
              if TestName == 'ClassicalSchar01':
                     # Reproduction of the Classical Schar case (one solve)
                     self.solType = {'StaticSolve': True, 'NLTranSolve': False, 'HermFuncGrid': True, \
                                'DynSGS': False, 'SolveFull': False, 'SolveSchur': True, \
                                'ToRestart': True, 'IsRestart': False, 'NewtonLin': True, \
                                'Smooth3Layer': False, 'UnifStrat': True, 'ExactBC': False, \
                                'UnifWind': True, 'LinShear': False, 'MakePlots': True}
                            
                     self.setUserData(191, 86, 70.0, 22.0, 280.0, 
                                      7000.0, 10000.0, 1.0, \
                                      250.0, 0.01, 0.0065, 0.003, 2, 1.2E+4, 'uwpt_static')
                     
              elif TestName == 'ClassicalScharIter':
                     # Newton iteration with Classical Schar as initial guess
                     self.solType = {'StaticSolve': True, 'NLTranSolve': False, 'HermFuncGrid': True, \
                                'DynSGS': False, 'SolveFull': False, 'SolveSchur': True, \
                                'ToRestart': True, 'IsRestart': False, 'NewtonLin': True, \
                                'Smooth3Layer': False, 'UnifStrat': True, 'ExactBC': True, \
                                'UnifWind': True, 'LinShear': False, 'MakePlots': True}
                            
                     self.setUserData(191, 86, 70.0, 22.0, 280.0, \
                                      7000.0, 10000.0, 1.0, \
                                      25.0, 0.01, 0.0065, 0.003, 2, 1.2E+4, 'uwpt_static')
                     
              elif TestName == 'UniformStratStatic':
                     # Newton iteration with smooth stratification
                     self.solType = {'StaticSolve': True, 'NLTranSolve': False, 'HermFuncGrid': True, \
                                'DynSGS': False, 'SolveFull': False, 'SolveSchur': True, \
                                'ToRestart': True, 'IsRestart': False, 'NewtonLin': True,\
                                'Smooth3Layer': False, 'UnifStrat': True, 'ExactBC': True, \
                                'UnifWind': False, 'LinShear': False, 'MakePlots': True}
                            
                     self.setUserData(191, 96, 75.0, 32.0, 300.0, \
                                      7000.0, 10000.0, 1.0, \
                                      25.0, 0.01, 0.0065, 0.003, 3, 1.2E+4, 'uwpt_static') 
                            
              elif TestName == 'DiscreteStratStatic':
                     # Newton iteration with discrete stratification
                     self.solType = {'StaticSolve': True, 'NLTranSolve': False, 'HermFuncGrid': True, \
                                'DynSGS': False, 'SolveFull': False, 'SolveSchur': True, \
                                'ToRestart': True, 'IsRestart': False, 'NewtonLin': True, \
                                'Smooth3Layer': True, 'UnifStrat': False, 'ExactBC': True, \
                                'UnifWind': False, 'LinShear': False, 'MakePlots': True}
                            
                     self.setUserData(191, 96, 75.0, 32.0, 300.0, \
                                      7000.0, 10000.0, 1.0, \
                                      25.0, 0.01, 0.0065, 0.003, 3, 1.2E+4, 'uwpt_static')
              
              elif TestName == "UniformTestTransient":
                     # Wave breaking in uniform stratification
                     self.solType = {'StaticSolve': False, 'NLTranSolve': True, 'HermFuncGrid': False, \
                                'DynSGS': True, 'SolveFull': False, 'SolveSchur': True, \
                                'ToRestart': True, 'IsRestart': False, 'NewtonLin': True, \
                                'Smooth3Layer': False, 'UnifStrat': True, 'ExactBC': True, \
                                'UnifWind': False, 'LinShear': False, 'MakePlots': False}
                            
                     # STRATIFICATION BY TEMPERATURE SOUNDING
                     self.setUserData(683, 84, 150.0, 35.0, 300.0, \
                                      10000.0, 25000.0, 10.0,
                                      4000.0, 0.01, 0.0065, 0.002, 3, 1.5E+4, 'uwpt_transient')
              
              elif TestName == "3LayerTestTransient":
                     # Wave breaking in 3 layer stratified atmosphere
                     self.solType = {'StaticSolve': False, 'NLTranSolve': True, 'HermFuncGrid': True, \
                                'DynSGS': True, 'SolveFull': False, 'SolveSchur': True, \
                                'ToRestart': True, 'IsRestart': False, 'NewtonLin': True, \
                                'Smooth3Layer': True, 'UnifStrat': False, 'ExactBC': True, \
                                'UnifWind': False, 'LinShear': False, 'MakePlots': True}
                            
                     # STRATIFICATION BY TEMPERATURE SOUNDING
                     self.setUserData(683, 98, 150.0, 42.0, 300.0, \
                                      10000.0, 25000.0, 10.0, \
                                      4000.0, 0.01, 0.0065, 0.002, 3, 1.5E+4, 'uwpt_transient')
              
              else:
                     print('INVALID/UNIMPLEMENTED TEST CASE CONFIGURATION!')
                     
       def setUserData(self, NX, NZ, XF, ZF, T0, depth, width, rlf, h0, NBVF, GAMT, GAMS, Mountain, kC, latBC):
              
              # Set physical constants (dry air)
              gc = 9.80601
              P0 = 1.0E5
              cp = 1004.5
              Rd = 287.06
              Kp = Rd / cp
              cv = cp - Rd
              gam = cp / cv
              self.PHYS = (gc, P0, cp, Rd, Kp, cv, gam, NBVF)
              
              # Set grid dimensions and order
              L2 = 1.0E+3 * XF # In 10s of km
              L1 = -L2
              ZH = 1.0E+3 * ZF # In km
              OPS = (NX + 1) * (NZ + 1)
              iU = 0
              iW = 1
              iP = 2
              iT = 3
              self.varDex = [iU, iW, iP, iT]
              self.DIMS = [L1, L2, ZH, NX, NZ, OPS]
              
              # Background temperature profile
              self.Z_in = [0.0, 1.1E4, 2.5E4, ZH]
              TTP = T0 - GAMT * (self.Z_in[1] - self.Z_in[0])
              TH = TTP + GAMS * (self.Z_in[3] - self.Z_in[2])
              self.T_in = (T0, TTP, TTP, TH)
              
              # Background wind profil e
              self.JETOPS = (10.0, 16.822, 1.386, 15.0)
              
              # Set the Rayleigh options
              applyTop = True
              applyLateral = True
              mu = rlf * 1.0E-2
              rdex = [0, 1, 3]
              self.RLOPT = (depth, width, applyTop, applyLateral, mu, latBC, rdex)
              
              # Set the terrain options
              withWindow = False
              '''
              KAISER = 1 # Kaiser window profile
              SCHAR = 2 # Schar mountain profile nominal (Schar, 2001)
              EXPCOS = 3 # Even exponential and squared cosines product
              EXPPOL = 4 # Even exponential and even polynomial product
              INFILE = 5 # Data from a file (equally spaced points)
              '''
              if Mountain == 2:
                     aC = 5000.0
                     lC = 4000.0
              elif Mountain == 3:
                     aC = 5000.0
                     lC = 2.0 * mt.pi * 1.0E3
              else:
                     aC = 5000.0
                     lC = 4000.0
                     
              self.HOPT = [h0, aC, lC, kC, withWindow, Mountain]
              
              #% Transient solve parameters
              DT = 0.05 # seconds
              HR = 5.0 #/ 3600.0 # hours              
              DTF = 0.75 # scale time step              
              intMethodOrder = 4
              # 3rd or 4th order time integrator
              ET = HR * 60 * 60 # End time in seconds
              OTI = 5.0 # Time for diagnostic output
              ITI = 5.0 # Time for image output
              RTI = 1.0 # Time for time step and constant damping update
              
              self.TOPT = [DT, HR, DTF, intMethodOrder, ET, OTI, ITI, RTI]
       