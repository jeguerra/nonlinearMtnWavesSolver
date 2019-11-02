#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:05:20 2019

@author: jorge.guerra
"""
import time
import numpy as np
import scipy.linalg as dsl
#import scipy.sparse as sps
#import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.sparse.linalg as spl
import computeEulerEquationsLogPLogT as tendency

def computeIterativeSolveNL(PHYS, REFS, REFG, DX, DZ, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex, sysDex):
       lastSol = SOLT[:,0]
       
       def computeRHSUpdate(sol):
              fields, U, RdT = tendency.computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, U, RdT, botdex, topdex)
              rhs += tendency.computeRayleighTendency(REFG, fields, udex, wdex, pdex, tdex, botdex, topdex)
       
              return rhs
       
       def computeJacVecUpdate(sol, vec):
              fields, U, RdT = tendency.computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              jv = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, U, RdT, botdex, topdex)
              jv += tendency.computeRayleighTendency(REFG, fields, udex, wdex, pdex, tdex, botdex, topdex)
       
              return jv
              
       
       # Solve for nonlinear equilibrium (default krylov)
       jac_options = {'method':'gmres','inner_maxiter':5000,'outer_k':5}
       sol, info = opt.nonlin.nonlin_solve(computeRHSUpdate, lastSol, 
                                  jacobian=opt.nonlin.KrylovJacobian(**jac_options),
                                  iter=5, verbose=True,
                                  maxiter=100,
                                  line_search='armijo',
                                  full_output=True,
                                  raise_exception=False)
       
       # Solve for nonlinear equilibrium (modified direct krylov)
       '''
       jac_options = {'jdv':computeJacVecUpdate, 'method':'gmres','inner_maxiter':4000,'outer_k':10}
       sol, info = opt.nonlin.nonlin_solve(computeRHSUpdate, lastSol, 
                                  jacobian=opt.nonlin.KrylovJacobian(**jac_options),
                                  iter=5, verbose=True,
                                  maxiter=100,
                                  line_search='armijo',
                                  full_output=True,
                                  raise_exception=False)
       '''
       print('NL solver exit on: ', info)
       
       return sol

#------------------------------------------------------------------------------
# Iterative/Krylov un-approximated Jacobians
#------------------------------------------------------------------------------

class KrylovDirectJacobian(opt.nonlin.Jacobian):
    r"""
    Find a root of a function, using Krylov approximation for inverse Jacobian.

    This method is suitable for solving large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    rdiff : float, optional
        Relative step size to use in numerical differentiation.
    method : {'lgmres', 'gmres', 'bicgstab', 'cgs', 'minres'} or function
        Krylov method to use to approximate the Jacobian.
        Can be a string, or a function implementing the same interface as
        the iterative solvers in `scipy.sparse.linalg`.

        The default is `scipy.sparse.linalg.lgmres`.
    inner_M : LinearOperator or InverseJacobian
        Preconditioner for the inner Krylov iteration.
        Note that you can use also inverse Jacobians as (adaptive)
        preconditioners. For example,

        >>> from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
        >>> from scipy.optimize.nonlin import InverseJacobian
        >>> jac = BroydenFirst()
        >>> kjac = KrylovJacobian(inner_M=InverseJacobian(jac))

        If the preconditioner has a method named 'update', it will be called
        as ``update(x, f)`` after each nonlinear step, with ``x`` giving
        the current point, and ``f`` the current function value.
    inner_tol, inner_maxiter, ...
        Parameters to pass on to the \"inner\" Krylov solver.
        See `scipy.sparse.linalg.gmres` for details.
    outer_k : int, optional
        Size of the subspace kept across LGMRES nonlinear iterations.
        See `scipy.sparse.linalg.lgmres` for details.
    %(params_extra)s

    See Also
    --------
    scipy.sparse.linalg.gmres
    scipy.sparse.linalg.lgmres

    Notes
    -----
    This function implements a Newton-Krylov solver. The basic idea is
    to compute the inverse of the Jacobian with an iterative Krylov
    method. These methods require only evaluating the Jacobian-vector
    products, which are NOT approximated by evaluated directly and
    passed to the "matvec" member function.

    Due to the use of iterative matrix inverses, these methods can
    deal with large nonlinear problems.

    SciPy's `scipy.sparse.linalg` module offers a selection of Krylov
    solvers to choose from. The default here is `lgmres`, which is a
    variant of restarted GMRES iteration that reuses some of the
    information obtained in the previous Newton steps to invert
    Jacobians in subsequent steps.

    For a review on Newton-Krylov methods, see for example [1]_,
    and for the LGMRES sparse inverse method, see [2]_.

    References
    ----------
    .. [1] D.A. Knoll and D.E. Keyes, J. Comp. Phys. 193, 357 (2004).
           :doi:`10.1016/j.jcp.2003.08.010`
    .. [2] A.H. Baker and E.R. Jessup and T. Manteuffel,
           SIAM J. Matrix Anal. Appl. 26, 962 (2005).
           :doi:`10.1137/S0895479803422014`

    """

    def __init__(self, jdv=None, method='lgmres', inner_maxiter=20,
                 inner_M=None, outer_k=10, **kw):
        self.jac_vec = jdv
        self.preconditioner = inner_M
        self.method = dict(
            bicgstab = spl.bicgstab,
            gmres = spl.gmres,
            lgmres = spl.lgmres,
            cgs = spl.cgs,
            minres = spl.minres,
            ).get(method, method)

        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)

        if self.method is spl.gmres:
            # Replace GMRES's outer iteration with Newton steps
            self.method_kw['restrt'] = inner_maxiter
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('atol', 0)
        elif self.method is spl.gcrotmk:
            self.method_kw.setdefault('atol', 0)
        elif self.method is spl.lgmres:
            self.method_kw['outer_k'] = outer_k
            # Replace LGMRES's outer iteration with Newton steps
            self.method_kw['maxiter'] = 1
            # Carry LGMRES's `outer_v` vectors across nonlinear iterations
            self.method_kw.setdefault('outer_v', [])
            self.method_kw.setdefault('prepend_outer_v', True)
            # But don't carry the corresponding Jacobian*v products, in case
            # the Jacobian changes a lot in the nonlinear step
            #
            # XXX: some trust-region inspired ideas might be more efficient...
            #      See eg. Brown & Saad. But needs to be implemented separately
            #      since it's not an inexact Newton method.
            self.method_kw.setdefault('store_outer_Av', False)
            self.method_kw.setdefault('atol', 0)

        for key, value in kw.items():
            if not key.startswith('inner_'):
                raise ValueError("Unknown parameter %s" % key)
            self.method_kw[key[6:]] = value

    def matvec(self, v):
        nv = dsl.norm(v)
        if nv == 0:
            return 0*v
        # Compute the updated Jacobian-vector product
        r = self.jac_vec(self.x0, v)
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs, tol=0):
        if 'tol' in self.method_kw:
            sol, info = self.method(self.op, rhs, **self.method_kw)
        else:
            sol, info = self.method(self.op, rhs, tol=tol, **self.method_kw)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f

    def setup(self, x, f, func):
        opt.nonlin.Jacobian.setup(self, x, f, func)
        self.x0 = x
        self.f0 = f
        self.op = spl.aslinearoperator(self)