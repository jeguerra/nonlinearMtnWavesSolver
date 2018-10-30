function [REFS, DOPS] = computeGridRefState_LogPLogTh(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL)
    %% Compute the Hermite and Legendre points and derivatives for this grid
    % Set the boundary indices and operator dimension
    OPS = NX*NZ;
    
    % Set the domain scale
    dscale = 0.5 * DS.L;
    %
    %% Get the Hermite Function derivative matrix and grid (ALTERNATE METHOD)
    [xo,~] = herdif(NX, 2, dscale, false);
    [xh,~] = herdif(NX, 2, dscale, true);

    [~,~,w]=hegs(NX);
    W = spdiags(w, 0, NX, NX);

    [~, HT] = hefunm(NX-1, xo);
    [~, HTD] = hefunm(NX, xo);
    
    %% Compute the coefficients of spectral derivative in matrix form
    SDIFF = zeros(NX+1,NX);
    SDIFF(1,2) = sqrt(0.5);
    SDIFF(NX + 1,NX) = -sqrt(NX * 0.5);
    SDIFF(NX,NX-1) = -sqrt((NX - 1) * 0.5);

    for cc = NX-2:-1:1
        SDIFF(cc+1,cc+2) = sqrt((cc + 1) * 0.5);
        SDIFF(cc+1,cc) = -sqrt(cc * 0.5);
    end

    b = max(xo) / dscale;
    DDX_H = b * HTD' * SDIFF * (HT * W);
    %}
    % Get the Hermite derivative matrix and grid (BAD DEFAULT NX > 240)
    %[xh,DDX_H] = herdif(NX, 1, dscale, true);
    % Get the Chebyshev nodes and compute the vertical derivative matrix
    [zlc, ~] = chebdif(NZ, 1);
    zl = DS.zH * 0.5 * (zlc + 1.0);
    zlc = 0.5 * (zlc + 1.0);
    alpha = exp(-0.5 * zlc);
    beta = (-0.5) * ones(size(zlc'));
    %DDZ_L = (1.0 / DS.zH) * poldif(zlc, 1);
    DDZ_L = (1.0 / DS.zH) * poldif(zlc, alpha, beta);

    %% Compute the terrain and derivatives
    [ht,dhdx] = computeTopoDerivative(TestCase, xh, DS, RAY);
    
    %% XZ grid for Legendre nodes in the vertical
    [HTZL,~] = meshgrid(ht,zl);
    [XL,ZL] = meshgrid(xh,zl);
  
    %% Gal-Chen, Sommerville coordinate
    %{
    dzdh = (1.0 - ZL / DS.zH);
    dxidz = (DS.zH - HTZL);
    sigma = DS.zH * dxidz.^(-1);
    %}
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
    RL = reshape(rayField,OPS,1);

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
    %dthref = exp(dlthref);
    lthref = 1.0 / BS.gam * lpref - lrref + ...
        BS.Rd / BS.cp * log(BS.p0) - log(BS.Rd);
    thref = exp(lthref);
    thref0 = min(min(thref));

%{
    %% Plot background fields including mean Ri number
    figure;
    subplot(2,2,1);
    plot(ujref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Speed $(m s^{-1})$');
    ylabel('Height (km)');
    ylim([0.0 35.0]);
    drawnow;
    
    subplot(2,2,2);
    plot(pref(:,1) ./ (rref(:,1) * BS.Rd),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    %title('Temperature Profile','FontSize',30);
    xlabel('Temperature (K)');
    ylabel('Height (km)');
    ylim([0.0 35.0]);
    drawnow;
    
    subplot(2,2,3);
    plot(thref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Potential Temperature (K)');
    ylabel('Height (km)');
    ylim([0.0 35.0]);
    drawnow;
    
    subplot(2,2,4);
    plot(sqrt(BS.ga * dlthref(:,1)),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Brunt-V\"ais\"ala Frequency ($s^{-1}$)');
    ylabel('Height (km)');
    ylim([0.0 35.0]);
    drawnow;
    
    export_fig('BACKGROUND_PROFILES.png');
    
    figure;
    plot(dujref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Shear $(s^{-1})$');
    ylabel('Height (km)');
    ylim([0.0 35.0]);
    drawnow;
    pause;
%}
    
    REFS = struct('ujref',ujref,'dujref',dujref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'lthref',lthref,'dlthref',dlthref, ...
        'pref',pref,'rref',rref,'thref',thref,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ_L',DDZ_L, 'RL', RL, ...
        'DDX_H',DDX_H,'sigma',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);
    
    %% Unwrap the derivative matrices into operator for 2D implementation
    
    % Compute the vertical derivatives operator (Lagrange expansion)
    DDXI_OP = spalloc(OPS, OPS, REFS.NX *REFS.NZ^2);
    for cc=1:REFS.NX
        ddex = (1:REFS.NZ) + (cc - 1) * REFS.NZ;
        DDXI_OP(ddex,ddex) = REFS.DDZ_L;
    end

    % Compute the horizontal derivatives operator (Hermite Function expansion)
    DDA_OP = spalloc(OPS, OPS, REFS.NZ * REFS.NX^2);
    for rr=1:REFS.NZ
        ddex = (1:REFS.NZ:OPS) + (rr - 1);
        DDA_OP(ddex,ddex) = REFS.DDX_H;
    end
    
    %% Assemble the block global operator L
    SIGMA = spdiags(reshape(REFS.sigma,OPS,1), 0, OPS, OPS);
    U0 = spdiags(reshape(REFS.ujref,OPS,1), 0, OPS, OPS);
    DUDZ = spdiags(reshape(REFS.dujref,OPS,1), 0, OPS, OPS);
    DLPDZ = spdiags(reshape(REFS.dlpref,OPS,1), 0, OPS, OPS);
    DLRDZ = spdiags(reshape(REFS.dlrref,OPS,1), 0, OPS, OPS);
    DLPTDZ = (1.0 / BS.gam * DLPDZ - DLRDZ);
    POR = spdiags(reshape(REFS.pref ./ REFS.rref,OPS,1), 0, OPS, OPS);
    DX = DDA_OP;
    U0DX = U0 * DX;
    unit = spdiags(ones(OPS,1),0, OPS, OPS);
    RAYM = spdiags(REFS.RL,0, OPS, OPS);
    
    % Horizontal momentum LHS
    LD11 = U0DX + RAY.nu1 * RAYM;
    LD12 = DUDZ;
    LD13 = POR * DX;
    %LD14 = ZSPR;
    % Vertical momentum LHS
    %LD21 = ZSPR;
    LD22 = U0DX + RAY.nu2 * RAYM;
    LD23 = POR * SIGMA * DDXI_OP + BS.ga * (1.0 / BS.gam - 1.0) * unit;
    LD24 = - BS.ga * unit;
    % Continuity (log pressure) LHS
    LD31 = BS.gam * DDA_OP;
    LD32 = BS.gam * SIGMA * DDXI_OP + DLPDZ;
    LD33 = U0DX + RAY.nu3 * RAYM;
    %LD34 = ZSPR;
    % Thermodynamic LHS
    %LD41 = ZSPR;
    LD42 = DLPTDZ;
    %LD43 = ZSPR;
    LD44 = U0DX + RAY.nu4 * RAYM;
    
    DOPS = struct('LD11', LD11, 'LD12', LD12, 'LD13', LD13, ...
                  'LD22', LD22, 'LD23', LD23, 'LD24', LD24, ...
                  'LD31', LD31, 'LD32', LD32, 'LD33', LD33, ...
                  'LD42', LD42, 'LD44', LD44);
  
end