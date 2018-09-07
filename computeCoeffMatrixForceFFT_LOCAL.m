function [LD,FF,REFS] = computeCoeffMatrixForceFFT(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL)
    % Set the boundary indices and operator dimension
    OPS = NX*NZ;
    
    %% Compute the Fourier space and vertical derivative matrix
    [~,DDX_H] = herdif(NX, 1, 0.5 * DS.L, true);
    x = linspace(DS.l1, DS.l2, NX+1);
    xh = (x(1:NX))';
    kx = (2*pi / DS.L) * [0:NX/2-1 -NX/2:-1];
    kx(1) = 1.0E-12;

    [zlc, ~] = chebdif(NZ, 1);
    zl = DS.zH * 0.5 * (zlc + 1.0);
    zlc = 0.5 * (zlc + 1.0);
    alpha = exp(-0.5 * zlc);
    beta = (-0.5) * ones(size(zlc'));
    %DDZ_L = (1.0 / DS.zH) * poldif(zlc, 1);
    DDZ_L = (1.0 / DS.zH) * poldif(zlc, alpha, beta);
       
    %% Compute the terrain and derivatives
    [ht,~] = computeTopoDerivative(TestCase,xh,DS,RAY);

    %% XZ grid for Legendre nodes in the vertical
    [HTZL,~] = meshgrid(ht,zl);
    [XL,ZL] = meshgrid(xh,zl);
    [KF,ZKL] = meshgrid(kx,zl);
  
    %% High Order Improved Guellrich coordinate
    % 3 parameter function
    xi = ZL / DS.zH;
    ang = 0.5 * pi * xi;
    AR = 1.0E-3;
    p = 20;
    q = 5;
    fxi = exp(-p/q * xi) .* cos(ang).^p + AR * xi .* (1.0 - xi);
    dfdxi = -p/q * exp(-p/q * xi) .* cos(ang).^p ...
            -(0.5 * p) * pi * exp(-p/q * xi) .* sin(ang) .* cos(ang).^(p-1) ...
            -AR * (1.0 - 2 * xi);
    dzdh = fxi;
    dxidz = DS.zH + HTZL .* (dfdxi - fxi);
    sigma = DS.zH * dxidz.^(-1);
    %}
    % Adjust Z with terrain following coords
    ZTL = (dzdh .* HTZL) + ZL;
    
    %% Compute the terrain slope derivatives using FFT derivative in X
    DZT = 1i * KF .* fft(HTZL,NX,2);

    %% Compute the reference state initialization
    if strcmp(TestCase,'ShearJetSchar') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(BS, DS.zH, zl, ZTL, RAY);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
    elseif strcmp(TestCase,'ShearJetScharCBVF') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressureCBVF(BS, ZTL);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
    elseif strcmp(TestCase,'ClassicalSchar') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressureCBVF(BS, ZTL);
        [ujref,dujref] = computeJetProfileUniform(UJ, lpref);
    elseif strcmp(TestCase,'AndesMtn') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(BS, DS.zH, zl, ZTL, RAY);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
    end
    
    %% Compute the vertical profiles of density and pressure
    pref = exp(lpref);
    rref = exp(lrref);
    rref0 = max(max(rref));
    % Background potential temperature profile
    dlthref = 1.0 / BS.gam * dlpref - dlrref;
    lthref = 1.0 / BS.gam * lpref - lrref + ...
        BS.Rd / BS.cp * log(BS.p0) - log(BS.Rd);
    thref = exp(lthref);
    thref0 = min(min(thref));
    
    REFS = struct('ujref',ujref,'dujref',dujref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'lthref',lthref,'dlthref',dlthref,...
        'pref',pref,'rref',rref,'thref',thref,'KF',KF,'ZKL',ZKL,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ',DDZ_L, ...
        'DDX_H',DDX_H,'sigma',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);
    
    %% Compute the Rayleigh field
    [rayField, ~] = computeRayleighXZ(DS,1.0,RAY.depth,RAY.width,XL,ZL,applyTopRL,applyLateralRL);
    %[rayField, ~] = computeRayleighPolar(DS,1.0,RAY.depth,XL,ZL);
    %[rayField, ~] = computeRayleighEllipse(DS,1.0,RAY.depth,RAY.width,XL,ZL);
    RL = reshape(rayField,OPS,1);

    %% Unwrap the derivative matrices into operators onto a state 1D vector
    % Compute the vertical derivatives operator (Legendre expansion)
    DDXI_OP = spalloc(OPS, OPS, OPS);
    for cc=1:NX
        ddex = (1:NZ) + (cc - 1) * NZ;
        DDXI_OP(ddex,ddex) = DDZ_L;
    end
    %DDXI_OP = sparse(DDXI_OP);
    %spy(DDXI_OP); pause;

    %% Assemble the block global operator L
    U0 = spdiags(reshape(ujref,OPS,1), 0, OPS, OPS);
    KX = spdiags(reshape(KF,OPS,1), 0, OPS, OPS);
    KXU0 = KX .* U0; clear U0;
    DUDZ = spdiags(reshape(dujref,OPS,1), 0, OPS, OPS);
    DLPDZ = spdiags(reshape(dlpref,OPS,1), 0, OPS, OPS);
    DLRDZ = spdiags(reshape(dlrref,OPS,1), 0, OPS, OPS);
    DLPTDZ = (1.0 / BS.gam * DLPDZ - DLRDZ);
    POR = spdiags(reshape(pref ./ rref,OPS,1), 0,  OPS, OPS);
    unit = spdiags(ones(OPS,1),0, OPS, OPS);
    SIGMA = spdiags(reshape(sigma,OPS,1), 0, OPS, OPS);
    
    RAYF = spdiags(RL,0, OPS, OPS);
    % Horizontal momentum LHS
    LD11 = 1i * KXU0 + RAY.nu1 * RAYF;
    LD12 = DUDZ;
    LD13 = 1i * POR * KX;
    LD14 = sparse(OPS,OPS);
    % Vertical momentum LHS
    LD21 = sparse(OPS,OPS);
    LD22 = 1i * KXU0 + RAY.nu2 * RAYF;
    LD23 = POR * SIGMA * DDXI_OP + BS.ga * (1.0 / BS.gam - 1.0) * unit;
    LD24 = - BS.ga * unit;
    % Continuity LHS
    LD31 = BS.gam * 1i * KX;
    LD32 = BS.gam * SIGMA * DDXI_OP + DLPDZ;
    LD33 = 1i * KXU0 + RAY.nu3 * RAYF;
    LD34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    LD41 = sparse(OPS,OPS);
    LD42 = DLPTDZ;
    LD43 = sparse(OPS,OPS);
    LD44 = 1i * KXU0 + RAY.nu4 * RAYF;
    
    clear KX KXU0 POR DUDZ DLPDZ DLPTDZ;
    
    %{
    % Horizontal momentum LHS
    L11 = 1i * KXU0;
    L12 = sparse(OPS,OPS);
    L13 = 1i * POR * KX;
    L14 = sparse(OPS,OPS);
    % Vertical momentum LHS
    L21 = sparse(OPS,OPS);
    L22 = 1i * KXU0;
    L23 = POR * SIGMA * DDXI_OP;
    L24 = sparse(OPS,OPS);
    % Continuity LHS
    L31 = BS.gam * 1i * KX;
    L32 = BS.gam * SIGMA * DDXI_OP;
    L33 = 1i * KXU0;
    L34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    L41 = sparse(OPS,OPS);
    L42 = sparse(OPS,OPS);
    L43 = sparse(OPS,OPS);
    L44 = 1i * KXU0;
    
    clear KX KXU0 POR;

    %% Assemble the algebraic part (Rayleigh layer on the diagonal)
    % Horizontal momentum LHS
    B11 = sparse(OPS,OPS) + RAY.nu1 * (spdiags(RL,0, OPS, OPS));
    B12 = DUDZ;
    B13 = sparse(OPS,OPS);
    B14 = sparse(OPS,OPS);
    % Vertical momentum LHS
    B21 = sparse(OPS,OPS);
    B22 = sparse(OPS,OPS) + RAY.nu2 * (spdiags(RL,0, OPS, OPS));
    B23 = BS.ga * (1.0 / BS.gam - 1.0) * unit;
    B24 = -BS.ga * unit;
    % Continuity LHS
    B31 = sparse(OPS,OPS);
    B32 = DLPDZ;
    B33 = sparse(OPS,OPS) + RAY.nu3 * (spdiags(RL,0, OPS, OPS));
    B34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    B41 = sparse(OPS,OPS);
    B42 = DLPTDZ;
    B43 = sparse(OPS,OPS);
    B44 = sparse(OPS,OPS) + RAY.nu4 * spdiags(RL,0, OPS, OPS);
    
    clear DUDZ DLPDZ DLPTDZ;
    
    %% Assemble the left hand side operator
    LD11 = L11 + B11; clear L11 B11;
    LD12 = L12 + B12; clear L12 B12;
    LD13 = L13 + B13; clear L13 B13;
    LD14 = L14 + B14; clear L14 B14;

    LD21 = L21 + B21; clear L21 B21;
    LD22 = L22 + B22; clear L22 B22;
    LD23 = L23 + B23; clear L23 B23;
    LD24 = L24 + B24; clear L24 B24;

    LD31 = L31 + B31; clear L31 B31;
    LD32 = L32 + B32; clear L32 B32;
    LD33 = L33 + B33; clear L33 B33;
    LD34 = L34 + B34; clear L34 B34;

    LD41 = L41 + B41; clear L41 B41;
    LD42 = L42 + B42; clear L42 B42;
    LD43 = L43 + B43; clear L43 B43;
    LD44 = L44 + B44; clear L44 B44;
    %}

    %% Assemble the LHS operator (reorder u p w t)
    LD = [LD11 LD12 LD13 LD14 ; ...
          LD21 LD22 LD23 LD24 ; ...
          LD31 LD32 LD33 LD34 ; ...
          LD41 LD42 LD43 LD44];
      
    %% Assemble the force vector (reorder u p w t)
    F11 = zeros(OPS,1);
    F21 = zeros(OPS,1);
    F31 = zeros(OPS,1);
    F41 = zeros(OPS,1);
    FF = [F11 ; F21 ; F31 ; F41];
end