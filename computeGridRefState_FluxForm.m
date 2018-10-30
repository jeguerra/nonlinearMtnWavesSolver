function [REFS, DOPS] = computeGridRefState_FluxForm(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL)
    %% Compute the Hermite and Legendre points and derivatives for this grid
    % Set the boundary indices and operator dimension
    OPS = NX*NZ;
    % Set the horizontal domain scale
    dscale = 0.5 * DS.L;
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
    [ht,dhdx] = computeTopoDerivative(TestCase,xh,DS,RAY);
    
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
    % Background RhoTheta gradient
    dthref = thref .* dlthref;
    drref = rref .* dlrref;
%{
    %% Check plots background fields including mean Ri number
    fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
    subplot(2,2,1);
    plot(ujref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Speed (m s^{-1})','FontSize',30);
    ylabel('Altitude (km)','FontSize',30);
    ylim([0.0 35.0]);
    fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
    drawnow;
    
    subplot(2,2,2);
    plot(pref(:,1) ./ (rref(:,1) * BS.Rd),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    %title('Temperature Profile','FontSize',30);
    xlabel('Temperature (K)','FontSize',30);
    ylabel('Altitude (km)','FontSize',30);
    ylim([0.0 35.0]);
    fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
    drawnow;
    
    subplot(2,2,3);
    plot(0.01*pref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Pressure (hPa)','FontSize',30);
    ylabel('Altitude (km)','FontSize',30);
    ylim([0.0 35.0]);
    fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
    drawnow;
    
    subplot(2,2,4);
    plot(rref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Density (kg m^{-3})','FontSize',30);
    ylabel('Altitude (km)','FontSize',30);
    ylim([0.0 35.0]);
    fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
    drawnow;
    dirname = '../ShearJetSchar/';
    fname = [dirname 'BACKGROUND_PROFILES'];
    screen2png(fname);
    
    fig = figure('Position',[0 0 1200 1200]); fig.Color = 'w';
    plot(dujref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Shear (s^{-1})','FontSize',30);
    ylabel('Altitude (km)','FontSize',30);
    ylim([0.0 35.0]);
    fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
    drawnow;
    
    fig = figure('Position',[0 0 1200 1200]); fig.Color = 'w';
    Ri = -BS.ga * dlrref(:,1);
    Ri = Ri ./ (dujref(:,1).^2);
    semilogx(Ri,1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Ri','FontSize',30);
    ylabel('Altitude (km)','FontSize',30);
    ylim([0.0 20.0]);
    xlim([0.1 1.0E4]);
    fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
    drawnow;
    dirname = '../ShearJetSchar/';
    fname = [dirname 'RICHARDSON_NUMBER'];
    screen2png(fname);
    pause
%}
    
    REFS = struct('ujref',ujref,'dujref',dujref,'dthref',dthref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'lthref',lthref,'dlthref',dlthref, ...
        'pref',pref,'rref',rref,'thref',thref,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ_L',DDZ_L, 'RL', RL, ...
        'DDX_H',DDX_H,'sigma',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);

    %% Unwrap the derivative matrices into operator for 2D implementation
    
    % Compute the vertical derivatives operator (Lagrange expansion)
    DDXI_OP = spalloc(OPS, OPS, NZ^2);
    for cc=1:NX
        ddex = (1:NZ) + (cc - 1) * NZ;
        DDXI_OP(ddex,ddex) = DDZ_L;
    end

    % Compute the horizontal derivatives operator (Hermite Function expansion)
    DDX_OP = spalloc(OPS, OPS, NX^2);
    for rr=1:NZ
        ddex = (1:NZ:OPS) + (rr - 1);
        DDX_OP(ddex,ddex) = DDX_H;
    end
    
    %% Assemble the block global operator L
    SIGMA = spdiags(reshape(sigma,OPS,1), 0, OPS, OPS);
    U0 = spdiags(reshape(ujref,OPS,1), 0, OPS, OPS);
    DUDZ = spdiags(reshape(dujref,OPS,1), 0, OPS, OPS);
    DTHDZ = spdiags(reshape(dthref,OPS,1), 0, OPS, OPS);
    DLTHDZ = spdiags(reshape(dlthref,OPS,1), 0, OPS, OPS);
    THTZ = spdiags(reshape(thref,OPS,1), 0, OPS, OPS);
    ITHTZ = spdiags(reshape(thref.^(-1),OPS,1), 0, OPS, OPS);
    RDTZ = spdiags(reshape(pref ./ rref,OPS,1), 0, OPS, OPS);
    PGFTX = (BS.gam * RDTZ - U0.^2) * ITHTZ;
    PGFTZ = BS.gam * RDTZ * ITHTZ;
    U0DX = U0 * DDX_OP;
    unit = spdiags(ones(OPS,1),0, OPS, OPS);
    RAYM = spdiags(REFS.RL,0, OPS, OPS);
    
    % Horizontal momentum LHS
    LD11 = U0DX + RAY.nu1 * RAYM;
    LD12 = DUDZ - U0 * DLTHDZ;
    %LD13 = sparse(OPS,OPS);
    LD14 = PGFTX * DDX_OP;
    % Vertical momentum LHS
    %LD21 = sparse(OPS,OPS);
    LD22 = U0DX + RAY.nu2 * RAYM;
    LD23 = BS.ga * unit;
    LD24 = PGFTZ * SIGMA * DDXI_OP + BS.ga * (1.0 - BS.gam) * ITHTZ;
    % Continuity (density) LHS
    LD31 = DDX_OP;
    LD32 = SIGMA * DDXI_OP;
    LD33 = RAY.nu3 * RAYM;
    %LD34 = sparse(OPS,OPS);;
    % Thermodynamic LHS
    %LD41 = sparse(OPS,OPS);;
    LD42 = DTHDZ;
    LD43 = -THTZ * U0DX;
    LD44 = U0DX + RAY.nu4 * RAYM;
    
    DOPS = struct('LD11', LD11, 'LD12', LD12, 'LD14', LD14, ...
                  'LD22', LD22, 'LD23', LD23, 'LD24', LD24, ...
                  'LD31', LD31, 'LD32', LD32, 'LD33', LD33, ...
                  'LD42', LD42, 'LD43', LD43, 'LD44', LD44);
  
end