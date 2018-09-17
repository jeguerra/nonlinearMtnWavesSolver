function [LD,FF,REFS] = computeCoeffMatrixForceCBC_FluxForm(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL)
    %% Compute the Hermite and Legendre points and derivatives for this grid
    % Set the boundary indices and operator dimension
    OPS = NX*NZ;
    % Set the horizontal domain scale
    dscale = 0.5 * DS.L;
    % Get the Hermite Function derivative matrix and grid
    [xh,DDX_H] = herdif(NX, 1, dscale, true);
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
    thref = exp(1.0 / BS.gam * lpref - lrref + ...
        BS.Rd / BS.cp * log(BS.p0) - log(BS.Rd));
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
    
    REFS = struct('ujref',ujref,'dujref',dujref,'drref',drref,'dthref',dthref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'dlthref',dlthref, ...
        'pref',pref,'rref',rref,'thref',thref,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ',DDZ_L, ...
        'DDX_H',DDX_H,'sigma',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);

    %% Unwrap the derivative matrices into operator for 2D implementation
    
    % Compute the vertical derivatives operator (Lagrange expansion)
    % Compute the vertical derivatives operator (Lagrange expansion)
    DDXI_OP = spalloc(OPS, OPS, NX * NZ^2);
    for cc=1:NX
        ddex = (1:NZ) + (cc - 1) * NZ;
        DDXI_OP(ddex,ddex) = DDZ_L;
    end

    % Compute the horizontal derivatives operator (Hermite Function expansion)
    DDA_OP = spalloc(OPS, OPS, NZ * NX^2);
    for rr=1:NZ
        ddex = (1:NZ:OPS) + (rr - 1);
        DDA_OP(ddex,ddex) = DDX_H;
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
    U0DA = U0 * DDA_OP;
    unit = spdiags(ones(OPS,1),0, OPS, OPS);
    
    RAYM = spdiags(RL,0, OPS, OPS);
    ZSPR = sparse(OPS,OPS);
    % Horizontal momentum LHS
    L11 = U0DA;
    L12 = ZSPR;
    L13 = ZSPR;
    L14 = PGFTX * DDA_OP;
    % Vertical momentum LHS
    L21 = ZSPR;
    L22 = U0DA;
    L23 = ZSPR;
    L24 = PGFTZ * SIGMA * DDXI_OP;
    % Continuity LHS
    L31 = DDA_OP;
    L32 = SIGMA * DDXI_OP;
    L33 = ZSPR;
    L34 = ZSPR;
    % Thermodynamic LHS
    L41 = ZSPR;
    L42 = ZSPR;
    L43 = -THTZ * U0DA;
    L44 = U0DA;

    %% Assemble the algebraic part (Rayleigh layer on the diagonal)
    % Horizontal momentum LHS
    B11 = RAY.nu1 * RAYM;
    B12 = DUDZ - U0 * DLTHDZ;
    B13 = ZSPR;
    B14 = ZSPR;
    % Vertical momentum LHS
    B21 = ZSPR;
    B22 = RAY.nu2 * RAYM;
    B23 = BS.ga * unit;
    B24 = BS.ga * (1.0 - BS.gam) * ITHTZ;
    % Continuity LHS (using density weighted change of variable in W)
    B31 = ZSPR;
    B32 = ZSPR;
    B33 = RAY.nu3 * RAYM;
    B34 = ZSPR;
    % Thermodynamic LHS
    B41 = ZSPR;
    B42 = DTHDZ;
    B43 = ZSPR;
    B44 = RAY.nu4 * RAYM;
    
    %% Assemble the left hand side operator
    LD11 = L11 + B11;
    LD12 = L12 + B12;
    LD13 = L13 + B13;
    LD14 = L14 + B14;

    LD21 = L21 + B21;
    LD22 = L22 + B22;
    LD23 = L23 + B23;
    LD24 = L24 + B24;

    LD31 = L31 + B31;
    LD32 = L32 + B32;
    LD33 = L33 + B33;
    LD34 = L34 + B34;

    LD41 = L41 + B41;
    LD42 = L42 + B42;
    LD43 = L43 + B43;
    LD44 = L44 + B44;
    
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