function [LD,FF,REFS] = computeCoeffMatrixForceFFT(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL)
    %% Compute the Hermite and Legendre points and derivatives for this grid
    [~,DDX_H] = herdif(NX, 1, 0.5 * DS.L, true);
    x = linspace(DS.l1, DS.l2, NX+1);
    xh = (x(1:NX))';
    kx = (2*pi / DS.L) * [0:NX/2-1 -NX/2:-1];
    kx(1) = 1.0E-10;

    [zlc,~] = chebdif(NZ,1);
    zl = DS.zH * 0.5 * (zlc + 1.0);
    zlc = 0.5 * (zlc + 1.0);
    DDZ_L = (1.0 / DS.zH) * poldif(zlc, 1);
       
    %% Compute the terrain and derivatives
    [ht,~] = computeTopoDerivative(TestCase,xh,DS,RAY);

    %% XZ grid for Legendre nodes in the vertical
    [HTZL,~] = meshgrid(ht,zl);
    [XL,ZL] = meshgrid(xh,zl);
    [KF,~] = meshgrid(kx,zl);
  
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
    
    %% Make the matrix convolution operator for BC... VERY EXPENSIVE
    DTP = sparse(convmtx(DZT(1,:)', NX));

    %% Compute the reference state initialization
    if strcmp(TestCase,'ShearJetSchar') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(BS, DS.zH, zl, ZTL, RAY);
        %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
        [lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, DS.zH, zl, ZL, RAY);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
    elseif strcmp(TestCase,'ShearJetScharCBVF') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressureCBVF(BS, ZTL);
        [lprefU,~,dlprefU,~] = computeBackgroundPressureCBVF(BS, ZL);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
    elseif strcmp(TestCase,'ClassicalSchar') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressureCBVF(BS, ZTL);
        [ujref,dujref] = computeJetProfileUniform(UJ, lpref);
    elseif strcmp(TestCase,'AndesMtn') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(BS, DS.zH, zl, ZTL, RAY);
        [lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, DS.zH, zl, ZL, RAY);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
    end
    
    %% Compute the vertical profiles of density and pressure
    pref = exp(lpref);
    rref = exp(lrref);
    por = pref ./ rref;
    rref0 = max(max(rref));
    % Background potential temperature profile
    dlthref = 1.0 / BS.gam * dlpref - dlrref;
    thref = exp(1.0 / BS.gam * lpref - lrref + ...
        BS.Rd / BS.cp * log(BS.p0) - log(BS.Rd));
    thref0 = min(min(thref));
    dlujref = dujref ./ ujref;
    
    REFS = struct('ujref',ujref,'dujref',dujref,'dlujref',dlujref,'por',por, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'dlthref',dlthref,...
        'pref',pref,'rref',rref,'thref',thref,'KF',KF,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ',DDZ_L, ...
        'DDX_H',DDX_H,'sigma',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);
    
    %% Compute the Rayleigh field
    [rayField, ~] = computeRayleighXZ(DS,1.0,RAY.depth,RAY.width,XL,ZL,applyTopRL,applyLateralRL);
    %[rayField, ~] = computeRayleighPolar(DS,1.0,RAY.depth,XL,ZL);
    %[rayField, ~] = computeRayleighEllipse(DS,1.0,RAY.depth,RAY.width,XL,ZL);
    RL = reshape(rayField,NX*NZ,1);

    %% Unwrap the derivative matrices into operators onto a state 1D vector
    % Compute the vertical derivatives operator (Legendre expansion)
    DDXI_OP = zeros(NX*NZ);
    for cc=1:NX
        ddex = (1:NZ) + (cc - 1) * NZ;
        DDXI_OP(ddex,ddex) = DDZ_L;
    end
    DDXI_OP = sparse(DDXI_OP);
%{
    % Compute the horizontal derivatives operator (Hermite expansion)
    DDX_OP = zeros(NX*NZ);
    for rr=1:NZ
        ddex = (1:NZ:NX*NZ) + (rr - 1);
        DDX_OP(ddex,ddex) = DDX_H;
    end
    DDX_OP = sparse(DDX_OP);
%}    
    %% Assemble the block global operator L
    OPS = NX*NZ;
    U0 = spdiags(reshape(ujref,OPS,1), 0, OPS, OPS);
    IU0 = spdiags(reshape(ujref.^(-1),OPS,1), 0, OPS, OPS);
    KX = spdiags(reshape(KF,OPS,1), 0, OPS, OPS);
    DUDZ = spdiags(reshape(dujref,OPS,1), 0, OPS, OPS);
    DLPDZ = spdiags(reshape(dlpref,OPS,1), 0, OPS, OPS);
    DLRDZ = spdiags(reshape(dlrref,OPS,1), 0, OPS, OPS);
    POR = spdiags(reshape(por,OPS,1), 0,  OPS, OPS);
    %U0DX = U0 * DDX_OP;
    unit = spdiags(ones(OPS,1),0, OPS, OPS);
    SIGMA = spdiags(reshape(sigma,OPS,1), 0, OPS, OPS);

    % Horizontal momentum LHS
    %L11 = 1i * KX .* U0;
    %L12 = sparse(OPS,OPS);
    %L13 = sparse(OPS,OPS);
    %L14 = 1i * POR * KX;
    % Vertical momentum LHS
    %L21 = sparse(OPS,OPS);
    L22 = 1i * KX .* U0;
    L23 = sparse(OPS,OPS);
    L24 = POR * SIGMA * DDXI_OP;
    % Continuity LHS
    %L31 = 1i * KX;
    L32 = SIGMA * DDXI_OP;
    L33 = 1i * KX .* U0;
    L34 = - 1i * POR * IU0 * KX;
    % Thermodynamic LHS
    %L41 = sparse(OPS,OPS);
    L42 = sparse(OPS,OPS);
    L43 = - 1i * KX .* U0;
    L44 = 1.0 / BS.gam * 1i * KX .* U0;

    %% Assemble the algebraic part (Rayleigh layer on the diagonal)
    % Horizontal momentum LHS
    %B11 = sparse(OPS,OPS) + RAY.nu1 * (spdiags(RL,0, OPS, OPS));% + RTP_OP);
    %B12 = DUDZ;
    %B13 = sparse(OPS,OPS);
    %B14 = sparse(OPS,OPS);
    % Vertical momentum LHS
    %B21 = sparse(OPS,OPS);
    B22 = sparse(OPS,OPS) + RAY.nu2 * (spdiags(RL,0, OPS, OPS));% + RTP_OP);
    B23 = BS.ga * unit;
    B24 = -BS.ga * unit;
    % Continuity LHS
    %B31 = sparse(OPS,OPS);
    B32 = DLRDZ - IU0 * DUDZ;
    B33 = sparse(OPS,OPS) + RAY.nu3 * (spdiags(RL,0, OPS, OPS));% + RTP_OP);
    B34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    %B41 = sparse(OPS,OPS);
    B42 = (1.0 / BS.gam * DLPDZ - DLRDZ);
    B43 = sparse(OPS,OPS) - RAY.nu4 * spdiags(RL,0, OPS, OPS);
    B44 = sparse(OPS,OPS) + 1.0 / BS.gam * RAY.nu4 * spdiags(RL,0, OPS, OPS);% + RTP_OP);

    %% Adjust the operator for BC on w only
    bdex = 1:NZ:OPS;
    %B21(bdex,bdex) = -DTP;
    %B21(bdex,bdex) = sparse(NX,NX);
    B22(bdex,bdex) = speye(NX,NX);
    B23(bdex,bdex) = sparse(NX,NX);
    B24(bdex,bdex) = sparse(NX,NX);
    
    %% Neumann Boundary Conditions TOP BOUNDARY TO INFINITY
    %
    tdex = NZ:NZ:OPS;
    mbc = 1.0 * sqrt(BS.ga * dlthref(NZ,1)) / ujref(NZ,1);
    UNIT = spdiags(ones(NX,1), 0, NX, NX);
    L22(tdex,tdex) = SIGMA(tdex,tdex) * DDXI_OP(tdex,tdex) + mbc * UNIT;
    L33(tdex,tdex) = SIGMA(tdex,tdex) * DDXI_OP(tdex,tdex) + mbc * UNIT;
    L44(tdex,tdex) = SIGMA(tdex,tdex) * DDXI_OP(tdex,tdex) + mbc * UNIT;
    %}
    %% Assemble the left hand side operator
    %LD11 = L11 + B11;
    %LD12 = L12 + B12;
    %LD13 = L13 + B13;
    %LD14 = L14 + B14;

    %LD21 = L21 + B21;
    LD22 = L22 + B22;
    LD23 = L23 + B23;
    LD24 = L24 + B24;

    %LD31 = L31 + B31;
    LD32 = L32 + B32;
    LD33 = L33 + B33;
    LD34 = L34 + B34;

    %LD41 = L41 + B41;
    LD42 = L42 + B42;
    LD43 = L43 + B43;
    LD44 = L44 + B44;

    LD = [ LD22 LD23 LD24 ; LD32 LD33 LD34 ; LD42 LD43 LD44];

    %% Assemble the force vector
    %F11 = zeros(OPS,1);
    F21 = zeros(OPS,1);
    F31 = zeros(OPS,1);
    F41 = zeros(OPS,1);
    
    %% Adjust the force vector for the coupled BC
    F21(bdex) = ujref(1,:) .* DZT(1,:);
    FF = [F21 ; F31 ; F41];
end