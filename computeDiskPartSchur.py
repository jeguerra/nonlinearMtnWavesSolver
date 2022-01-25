#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:07:29 2022

Implements the disk partitioned Schur complement direct solver

@author: jeguerra
"""

import shelve
import numpy as np
import scipy.linalg as dsl
import scipy.sparse as sps

# Store a matrix to disk in column wise chucks
def storeColumnChunks(MM, Mname, dbName):
       # Set up storage and store full array
       mdb = shelve.open(dbName, flag='n')
       # Get the number of cpus
       import multiprocessing as mtp
       NCPU = int(1.25 * mtp.cpu_count())
       # Partition CS into NCPU column wise chuncks
       NC = MM.shape[1] # Number of columns in MM
       RC = NC % NCPU # Remainder of columns when dividing by NCPU
       SC = int((NC - RC) / NCPU) # Number of columns in each chunk
       
       # Loop over NCPU column chunks and store
       cranges = []
       for cc in range(NCPU):
              cbegin  = cc * SC
              if cc < NCPU - 1:
                     crange = range(cbegin,cbegin + SC)
              elif cc == NCPU - 1:
                     crange = range(cbegin,cbegin + SC + RC)
              
              cranges.append(crange)
              mdb[Mname + str(cc)] = MM[:,crange]
              
       mdb.close()
              
       return NCPU, cranges

def computeSchurBlock(dbName, blockName):
       # Open the blocks database
       bdb = shelve.open(dbName, flag='r')
       
       if blockName == 'AS':
              SB = sps.bmat([[bdb['LDIA'], bdb['LNA'], bdb['LOA']], \
                             [bdb['LDA'], bdb['A'], bdb['B']], \
                             [bdb['LHA'], bdb['E'], bdb['F']]], format='csr')
       elif blockName == 'BS':
              SB = sps.bmat([[bdb['LPA'], bdb['LQAR']], \
                             [bdb['C'], bdb['D']], \
                             [bdb['G'], bdb['H']]], format='csr')
       elif blockName == 'CS':
              SB = sps.bmat([[bdb['LMA'], bdb['I'], bdb['J']], \
                             [bdb['LQAC'], bdb['N'], bdb['O']]], format='csr')
       elif blockName == 'DS':
              SB = sps.bmat([[bdb['K'], bdb['M']], \
                             [bdb['P'], bdb['Q']]], format='csr')
       else:
              print('INVALID SCHUR BLOCK NAME!')
              
       bdb.close()

       return SB.toarray()

def solveDiskPartSchur(localDir, schurName, f1, f2):
       
       print('Solving linear system by Schur Complement...')
       # Factor DS and compute the Schur Complement of DS
       DS = computeSchurBlock(schurName,'DS')
       factorDS = dsl.lu_factor(DS, overwrite_a=True, check_finite=False)
       del(DS)
       print('Factor D... DONE!')
       
       # Store factor_DS for a little bit...
       FDS = shelve.open(localDir + 'factorDS', flag='n', protocol=4)
       FDS['factorDS'] = factorDS
       FDS.close()
       print('Store LU factor of D... DONE!')
       
       # Compute f2_hat = DS^-1 * f2 and f1_hat
       BS = computeSchurBlock(schurName,'BS')
       f2_hat = dsl.lu_solve(factorDS, f2)
       f1_hat = f1 - BS.dot(f2_hat)
       del(f1)
       del(BS) 
       del(f2_hat)
       print('Compute modified force vectors... DONE!')
       
       # Get CS block and store in column chunks
       CS = computeSchurBlock(schurName, 'CS')
       fileCS = localDir + 'CS'
       NCPU, CS_cranges = storeColumnChunks(CS, 'CS', fileCS)
       print('Partition block C into chunks and store... DONE!')
       del(CS)
       
       # Get AS block and store in column chunks
       AS = computeSchurBlock(schurName, 'AS')
       fileAS = localDir + 'AS'
       NCPU, AS_cranges = storeColumnChunks(AS, 'AS', fileAS)
       print('Partition block A into chunks and store... DONE!')
       del(AS)
       
       # Loop over the chunks from disk
       #AS = computeSchurBlock(schurName, 'AS')
       BS = computeSchurBlock(schurName, 'BS')
       ASmdb = shelve.open(fileAS)
       CSmdb = shelve.open(fileCS, flag='r')
       print('Computing DS^-1 * CS in chunks: ', NCPU)
       for cc in range(NCPU):
              # Get CS chunk
              #CS_crange = CS_cranges[cc] 
              CS_chunk = CSmdb['CS' + str(cc)]
              
              DS_chunk = dsl.lu_solve(factorDS, CS_chunk, overwrite_b=True, check_finite=False) # LONG EXECUTION
              del(CS_chunk)
              
              # Get AS chunk
              #AS_crange = AS_cranges[cc] 
              AS_chunk = ASmdb['AS' + str(cc)]
              #AS[:,crange] -= BS.dot(DS_chunk) # LONG EXECUTION
              ASmdb['AS' + str(cc)] = AS_chunk - BS.dot(DS_chunk)
              del(AS_chunk)
              del(DS_chunk)
              print('Computed chunk: ', cc+1)
              
       CSmdb.close()
       del(BS)
       del(factorDS)
       
       # Reassemble Schur complement of DS from AS chunk storage
       print('Computing Schur Complement of D from chunks.')
       DS_SC = ASmdb['AS0']
       for cc in range(1,NCPU):
              DS_SC = np.hstack((DS_SC, ASmdb['AS' + str(cc)]))
       ASmdb.close()
       print('Solve DS^-1 * CS... DONE!')
       print('Compute Schur Complement of D... DONE!')
       #'''
       # Apply Schur C. solver on block partitioned DS_SC
       factorDS_SC = dsl.lu_factor(DS_SC, overwrite_a=True)
       del(DS_SC)
       print('Factor Schur Complement of D... DONE!')
       #'''
       sol1 = dsl.lu_solve(factorDS_SC, f1_hat, overwrite_b=True, check_finite=False)
       del(factorDS_SC)
       #sol1, icode = spl.bicgstab(AS, f1_hat)
       del(f1_hat)
       print('Solve for u and w... DONE!')
       
       CS = computeSchurBlock(schurName, 'CS')
       f2_hat = f2 - CS.dot(sol1)
       del(f2)
       del(CS)
       FDS = shelve.open(localDir + 'factorDS', flag='r', protocol=4)
       factorDS = FDS['factorDS']
       FDS.close()
       sol2 = dsl.lu_solve(factorDS, f2_hat, overwrite_b=True, check_finite=False)
       del(f2_hat)
       del(factorDS)
       print('Solve for ln(p) and ln(theta)... DONE!')
       dsol = np.concatenate((sol1, sol2))
       
       # Get memory back
       del(sol1); del(sol2)
       
       return dsol

