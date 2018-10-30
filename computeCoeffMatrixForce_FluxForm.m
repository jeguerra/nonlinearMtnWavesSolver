function [LD,FF] = computeCoeffMatrixForce_FluxForm(BS, RAY, REFS)
    %% Set the dimensions
    OPS = REFS.NX * REFS.NZ;
    
    %% Unwrap the derivative matrices into operator for 2D implementation
    
    % Compute the vertical derivatives operator (Lagrange expansion)
    DDXI_OP = spalloc(OPS, OPS, REFS.NZ^2);
    for cc=1:REFS.NX
        ddex = (1:REFS.NZ) + (cc - 1) * REFS.NZ;
        DDXI_OP(ddex,ddex) = REFS.DDZ_L;
    end

    % Compute the horizontal derivatives operator (Hermite Function expansion)
    DDX_OP = spalloc(OPS, OPS, REFS.NX^2);
    for rr=1:REFS.NZ
        ddex = (1:REFS.NZ:OPS) + (rr - 1);
        DDX_OP(ddex,ddex) = REFS.DDX_H;
    end

    %% Assemble the block global operator L
    SIGMA = spdiags(reshape(REFS.sigma,OPS,1), 0, OPS, OPS);
    U0 = spdiags(reshape(REFS.ujref,OPS,1), 0, OPS, OPS);
    DUDZ = spdiags(reshape(REFS.dujref,OPS,1), 0, OPS, OPS);
    DTHDZ = spdiags(reshape(REFS.dthref,OPS,1), 0, OPS, OPS);
    DLTHDZ = spdiags(reshape(REFS.dlthref,OPS,1), 0, OPS, OPS);
    THTZ = spdiags(reshape(REFS.thref,OPS,1), 0, OPS, OPS);
    ITHTZ = spdiags(reshape(REFS.thref.^(-1),OPS,1), 0, OPS, OPS);
    RDTZ = spdiags(reshape(REFS.pref ./ REFS.rref,OPS,1), 0, OPS, OPS);
    PGFTX = (BS.gam * RDTZ - U0.^2) * ITHTZ;
    PGFTZ = BS.gam * RDTZ * ITHTZ;
    U0DX = U0 * DDX_OP;
    unit = spdiags(ones(OPS,1),0, OPS, OPS);
    
    RAYM = spdiags(REFS.RL,0, OPS, OPS);
    ZSPR = sparse(OPS,OPS);
    % Horizontal momentum LHS
    LD11 = U0DX + RAY.nu1 * RAYM;
    LD12 = DUDZ - U0 * DLTHDZ;
    LD13 = ZSPR;
    LD14 = PGFTX * DDX_OP;
    % Vertical momentum LHS
    LD21 = ZSPR;
    LD22 = U0DX + RAY.nu2 * RAYM;
    LD23 = BS.ga * unit;
    LD24 = PGFTZ * SIGMA * DDXI_OP + BS.ga * (1.0 - BS.gam) * ITHTZ;
    % Continuity LHS
    LD31 = DDX_OP;
    LD32 = SIGMA * DDXI_OP;
    LD33 = ZSPR + RAY.nu3 * RAYM;
    LD34 = ZSPR;
    % Thermodynamic LHS
    LD41 = ZSPR;
    LD42 = DTHDZ;
    LD43 = -THTZ * U0DX;
    LD44 = U0DX + RAY.nu4 * RAYM;
    
    %% Assemble the LHS operator (reorder u rt r w)
    %
    LD = [LD11 LD12 LD13 LD14 ; ...
          LD21 LD22 LD23 LD24 ; ...
          LD31 LD32 LD33 LD34 ; ...
          LD41 LD42 LD43 LD44];
    
    %% Assemble the force vector (reorder u p w t)
    FF = zeros(4 * OPS,1);
end