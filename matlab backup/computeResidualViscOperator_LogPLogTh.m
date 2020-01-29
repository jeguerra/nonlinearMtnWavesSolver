function RVD = computeResidualViscOperator_LogPLogTh(REFS, RHS)
    %% Set the dimensions
    OPS = REFS.NX * REFS.NZ;
    
    %% Set the characteristic length scales (Nyquist length at the given order)
    DX = 2.0 * min(diff(REFS.XL(1,:)));
    DZ = 2.0 * min(diff(REFS.ZTL(:,1)));
    
    %% Fetch the derivative matrices into operator for 2D implementation
    
    % Compute the vertical derivatives operator (Lagrange expansion)
    DDZ_OP = REFS.DDZ_OP;

    % Compute the horizontal derivatives operator (Hermite Function expansion)
    DDX_OP = REFS.DDX_OP;
    
    %% Get the residual components
    %URES = abs(RHS(1:OPS)); UF = max(URES);
    %WRES = abs(RHS(OPS+1:2*OPS)); WF = max(WRES);
    %PRES = abs(RHS(2*OPS+1:3*OPS)); PF = max(PRES);
    TRES = abs(RHS(3*OPS+1:4*OPS)); 
    TF = REFS.thref0; 
    TRES = spdiags(TRES, 0);

    %% Assemble the block global operator RVD
    ZSPR = sparse(OPS,OPS);
    % Horizontal momentum residual viscosity
    %RVD11 = DDX_OP * (URES * DDX_OP) + DDZ_OP * (URES * DDZ_OP);
    RVD11 = ZSPR;
    % Vertical momentum residual viscosity
    %RVD22 = DDX_OP * (WRES * DDX_OP) + DDZ_OP * (WRES * DDZ_OP);
    RVD22 = ZSPR;
    % Continuity (log pressure) residual viscosity
    %RVD33 = DDX_OP * (PRES * DDX_OP) + DDZ_OP * (PRES * DDZ_OP);
    RVD33 = ZSPR;
    % Thermodynamic (log potential temp.) residual viscosity
    RVD44 = DX^2 * DDX_OP * (TRES * DDX_OP) + ...
            DZ^2 * DDZ_OP * (TRES * DDZ_OP);
    RVD44 = 2.0 * (1.0 / TF) * RVD44;
    %%RVD44 = ZSPR;

    %% Assemble the RVD operator
    %
    RVD = struct('RVD11',RVD11,'RVD22',RVD22,'RVD33',RVD33,'RVD44',RVD44);
end