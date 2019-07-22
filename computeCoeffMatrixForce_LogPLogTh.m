function [LD, FF] = computeCoeffMatrixForce_LogPLogTh(BS, RAY, REFS)
    %% Set the dimensions
    OPS = REFS.NX * REFS.NZ;
    
    %% Fetch the derivative matrices into operator for 2D implementation
    
    % Compute the vertical derivatives operator (Lagrange expansion)
    DDZ_OP = REFS.DDZ_OP;

    % Compute the horizontal derivatives operator (Hermite Function expansion)
    DDX_OP = REFS.DDX_OP;

    %% Assemble the block global operator L
    U0 = spdiags(reshape(REFS.ujref,OPS,1), 0, OPS, OPS);
    DUDZ = spdiags(reshape(REFS.dujref,OPS,1), 0, OPS, OPS);
    DLPDZ = spdiags(reshape(REFS.dlpref,OPS,1), 0, OPS, OPS);
    DLRDZ = spdiags(reshape(REFS.dlrref,OPS,1), 0, OPS, OPS);
    DLPTDZ = (1.0 / BS.gam * DLPDZ - DLRDZ);
    POR = spdiags(reshape(REFS.pref ./ REFS.rref,OPS,1), 0, OPS, OPS);
    U0DX = U0 * DDX_OP;
    unit = spdiags(ones(OPS,1),0, OPS, OPS);

    RAYM = spdiags(REFS.RL,0, OPS, OPS);
    ZSPR = sparse(OPS,OPS);
    % Horizontal momentum LHS
    LD11 = U0DX + RAY.nu1 * RAYM;
    LD12 = DUDZ;
    LD13 = POR * DDX_OP;
    LD14 = ZSPR;
    % Vertical momentum LHS
    LD21 = ZSPR;
    LD22 = U0DX + RAY.nu2 * RAYM;
    LD23 = POR * DDZ_OP + BS.ga * (1.0 / BS.gam - 1.0) * unit;
    LD24 = - BS.ga * unit;
    % Continuity (log pressure) LHS
    LD31 = BS.gam * DDX_OP;
    LD32 = BS.gam * DDZ_OP + DLPDZ;
    LD33 = U0DX + RAY.nu3 * RAYM;
    LD34 = ZSPR;
    % Thermodynamic LHS
    LD41 = ZSPR;
    LD42 = DLPTDZ;
    LD43 = ZSPR;
    LD44 = U0DX + RAY.nu4 * RAYM;

    %% Assemble the LHS operator (reorder u p w t)
    %
    LD = [LD11 LD12 LD13 LD14 ; ...
          LD21 LD22 LD23 LD24 ; ...
          LD31 LD32 LD33 LD34 ; ...
          LD41 LD42 LD43 LD44];
    
    %% Make the normal equations symmetric coefficient
    %NLD = [(LD11' * LD11) ZSPR (LD31' * LD31) ZSPR ; ...
    %       ZSPR (LD22' * LD22) (LD32' * LD32) (LD42' * LD42) ; ...
    %       (LD13' * LD13) (LD23' * LD23) (LD33' * LD33) ZSPR ; ...
    %       ZSPR (LD24' * LD24) ZSPR (LD44' * LD44)];
    
    %% Assemble the force vector (reorder u p w t)
    FF = zeros(4 * OPS,1);
end