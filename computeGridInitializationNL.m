function REFS = computeGridInitializationNL(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL)
    %% Compute the Hermite and Legendre points and derivatives for this grid
    
    % Set the domain scale
    dscale = 0.5 * DS.L;
    %{
    %% Use a truncated projection space
    [xo,~] = herdif(NX, 2, dscale, false);
    [xh,~] = herdif(NX, 2, dscale, true);

    [~,~,w]=hegs(NX);
    W = spdiags(w, 0, NX, NX);

    [~, HT] = hefunm(NXO-1, xo);
    [~, HTD] = hefunm(NXO, xo);
    
    %% Compute the coefficients of spectral derivative in matrix form
    SDIFF = zeros(NXO+1,NXO);
    SDIFF(1,2) = sqrt(0.5);
    SDIFF(NXO + 1,NXO) = -sqrt(NXO * 0.5);
    SDIFF(NXO,NXO-1) = -sqrt((NXO - 1) * 0.5);

    for cc = NXO-2:-1:1
        SDIFF(cc+1,cc+2) = sqrt((cc + 1) * 0.5);
        SDIFF(cc+1,cc) = -sqrt(cc * 0.5);
    end

    b = max(xo) / dscale;
    DDX_H = b * HTD' * SDIFF * (HT * W);
    %rcond(DDX_H)
    %rank(DDX_H)
    surf(DDX_H);
    figure;
    %}
    %
    [xh,DDX_H] = herdif(NX, 1, dscale, true);
    %}
    [zlc, ~] = chebdif(NZ, 1);
    zl = DS.zH * 0.5 * (zlc + 1.0);
    zlc = 0.5 * (zlc + 1.0);
    alpha = exp(-0.5 * zlc);
    beta = (-0.5) * ones(size(zlc'));
    %DDZ_L = (1.0 / DS.zH) * poldif(zlc, 1);
    DDZ_L = (1.0 / DS.zH) * poldif(zlc, alpha, beta);
    %}
           
    %% Compute the terrain and derivatives
    [ht,dhdx] = computeTopoDerivative(TestCase,xh,DS,RAY);
    
    %% XZ grid for Legendre nodes in the vertical
    [HTZL,~] = meshgrid(ht,zl);
    [XL,ZL] = meshgrid(xh,zl);
  
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
    % Make the global array of terrain derivative features
    DZT = ZTL;
    for rr=1:size(DZT,1)
        DZT(rr,:) = fxi(rr,:) .* dhdx';
    end
    
    %% Compute the Rayleigh field
    [rayField, ~] = computeRayleighXZ(DS,1.0,RAY.depth,RAY.width,XL,ZL,applyTopRL,applyLateralRL);
    %[rayField, ~] = computeRayleighPolar(DS,1.0,RAY.depth,XL,ZL);
    %[rayField, ~] = computeRayleighEllipse(DS,1.0,RAY.depth,RAY.width,XL,ZL);
    RL = reshape(rayField,NX*NZ,1);

    %% Compute the reference state initialization
    if strcmp(TestCase,'ShearJetSchar') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(BS, DS.zH, zl, ZTL, RAY);
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
    % Background potential temperature profile
    dlthref = 1.0 / BS.gam * dlpref - dlrref;
    thref = exp(1.0 / BS.gam * lpref - lrref + ...
        BS.Rd / BS.cp * log(BS.p0) - log(BS.Rd));
    thref0 = min(min(thref));
    % Background RhoTheta gradient
    dthref = thref .* dlthref;
    drthref = rref .* thref .* (dlrref + dlthref);
    
    REFS = struct('ujref',ujref,'dujref',dujref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'dlthref',dlthref, ...
        'pref',pref,'rref',rref,'thref',thref,'dthref',dthref,'drthref',drthref, ...
        'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT, ...
        'DDZ_L',DDZ_L,'DDX_H',DDX_H,'sigma',sigma,'RL',RL, ...
        'NX',NX,'NZ',NZ,'TestCase',TestCase,'thref0',thref0);

end