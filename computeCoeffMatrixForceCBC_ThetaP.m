function [LD,FF,REFS] = computeCoeffMatrixForceCBC_ThetaP(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL)
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
    rcond(DDX_H)
    rank(DDX_H)
    surf(DDX_H);
    figure;
    surf(QO);
    figure;
    %}
    %
    [xh,DDX_H] = herdif(NX, 1, dscale, true);
    % fix the diagonal
    %DDX_H(1:NX+1:end) = 1.0E-8 * ones(NX,1);
    %}
    [zlc, ~] = chebdif(NZ, 1);
    zl = DS.zH * 0.5 * (zlc + 1.0);
    zlc = 0.5 * (zlc + 1.0);
    DDZ_L = (1.0 / DS.zH) * poldif(zlc, 1);
    %}
    
    %[zl, DDZ_L] = lagdifJEG(NZ, 1, 2.0 * DS.zH);
       
    %% Compute the terrain and derivatives
    [ht,dhdx] = computeTopoDerivative(TestCase,xh,DS);
    
    %% XZ grid for Legendre nodes in the vertical
    [HTZL,~] = meshgrid(ht,zl);
    [XL,ZL] = meshgrid(xh,zl);
  
    %% Gal-Chen, Sommerville coordinate
    %{
    dzdh = (1.0 - ZL / DS.zH);
    dxidz = (DS.zH - HTZL);
    sigma = DS.zH * dxidz.^(-1);
    %}
    %% 8th Order Guellrich coordinate
    %{
    eta = ZL / DS.zH;
    ang = 0.5 * pi * eta;
    AR = 0.1;
    power = 8;
    fxi = cos(ang).^power + AR * eta;
    dfdxi = -(0.5 * power) * pi * sin(ang) .* cos(ang).^(power-1) + AR;
    dzdh = (1.0 - eta) .* fxi;
    dxidz = DS.zH + HTZL .* ((1.0 - eta) .* dfdxi - fxi);
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
        DZT(rr,:) = dhdx;
    end
    % Convert the slopes to alpha coordinate
    DZTA = DZT .* (1.0 + DZT.^2).^(-0.5);
    % Make the metric term for conversion to contravariant
    dxihatdx = sigma .* dzdh .* DZTA;
    
    %% Compute the Rayleigh field
    [rayField, ~] = computeRayleighXZ(DS,1.0,RAY.depth,RAY.width,XL,ZL,applyTopRL,applyLateralRL);
    %[rayField, ~] = computeRayleighPolar(DS,1.0,RAY.depth,XL,ZL);
    %[rayField, ~] = computeRayleighEllipse(DS,1.0,RAY.depth,RAY.width,XL,ZL);
    RL = reshape(rayField,NX*NZ,1);
    %BRV = reshape(BR,NX*NZ,1);

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
    rref0 = max(max(rref));
    % Background potential temperature profile
    LCN = BS.Rd / BS.cp * log(BS.p0) - log(BS.Rd);
    dlthref = 1.0 / BS.gam * dlpref - dlrref; 
    lthref = 1.0 / BS.gam * lpref - lrref + LCN;
    thref = exp(lthref);
    % Theta weighted change of variable
    thref0 = min(min(thref));
    rsc = sqrt(thref0) * thref.^(-0.5);
    %dthref = thref .* dlthref;

%{
    %% Plot background fields including mean Ri number
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
    plot(lthref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Log Potential Temperature (K)','FontSize',30);
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
    
    REFS = struct('ujref',ujref,'dujref',dujref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref, ...
        'pref',pref,'rref',rref,'thref',thref,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ',DDZ_L, ...
        'DDX_H',DDX_H,'sig',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);

    %% Unwrap the derivative matrices into operators onto a state 1D vector
    % Compute the vertical derivatives operator (Legendre expansion)
    DDZ_OP = zeros(NX*NZ);
    for cc=1:NX
        ddex = (1:NZ) + (cc - 1) * NZ;
        DDZ_OP(ddex,ddex) = DDZ_L;
    end
    DDZ_OP = sparse(DDZ_OP);

    % Compute the horizontal derivatives operator (Hermite expansion)
    DDX_OP = zeros(NX*NZ);
    for rr=1:NZ
        ddex = (1:NZ:NX*NZ) + (rr - 1);
        DDX_OP(ddex,ddex) = DDX_H;
    end
    DDX_OP = sparse(DDX_OP);

    %% Assemble the block global operator L
    OPS = NX*NZ;
    U0 = spdiags(reshape(ujref,OPS,1), 0, OPS, OPS);
    DU0DZ = spdiags(reshape(dujref,OPS,1), 0, OPS, OPS);
    DLPDZ = spdiags(reshape(dlpref,OPS,1), 0, OPS, OPS);
    DLTDZ = spdiags(reshape(dlthref,OPS,1), 0, OPS, OPS);
    POR = spdiags(reshape(pref ./ rref,OPS,1), 0,  OPS, OPS);
    %RSC = spdiags(reshape(rsc,OPS,1), 0, OPS, OPS);
    DXIHAT = spdiags(reshape(dxihatdx,OPS,1), 0, OPS, OPS);
    %U0DX = U0 * (DDX_OP - DXIHAT * DDZ_OP);
    U0DA = U0 * DDX_OP;
    unit = spdiags(ones(OPS,1),0, OPS, OPS);
    SIGMA = spdiags(reshape(sigma,OPS,1), 0, OPS, OPS);

    % Theta LHS
    L11 = U0DA;
    L12 = sparse(OPS,OPS);
    L13 = sparse(OPS,OPS);
    L14 = sparse(OPS,OPS);
    % Horizontal momentum LHS
    L21 = sparse(OPS,OPS);
    L22 = U0DA;
    L23 = POR * DDX_OP;
    L24 = sparse(OPS,OPS);
    % Pressure LHS
    L31 = sparse(OPS,OPS);
    L32 = BS.gam * DDX_OP;
    L33 = U0DA;
    L34 = BS.gam * SIGMA * DDZ_OP;
    % Vertical momentum LHS
    L41 = sparse(OPS,OPS);
    L42 = sparse(OPS,OPS);
    L43 = POR * SIGMA * DDZ_OP;
    L44 = U0DA;

    %% Assemble the algebraic part (Rayleigh layer on the diagonal)
    % Theta LHS
    B11 = sparse(OPS,OPS) + RAY.nu1 * spdiags(RL,0, OPS, OPS);
    B12 = -DXIHAT * DLTDZ;
    %B12 = sparse(OPS,OPS);
    B13 = sparse(OPS,OPS);
    B14 = SIGMA * DLTDZ;
    % Horizontal momentum LHS
    B21 = sparse(OPS,OPS);
    B22 = -DXIHAT * DU0DZ + RAY.nu2 * spdiags(RL,0, OPS, OPS);
    %B22 = sparse(OPS,OPS) + RAY.nu2 * spdiags(RL,0, OPS, OPS);
    B23 = sparse(OPS,OPS);
    B24 = SIGMA * DU0DZ;
    % Thermodynamic LHS
    B31 = sparse(OPS,OPS);
    B32 = -DXIHAT * DLPDZ;
    B33 = sparse(OPS,OPS) + RAY.nu3 * spdiags(RL,0, OPS, OPS);
    B34 = SIGMA * DLPDZ;
    % Vertical momentum LHS
    B41 = -BS.ga * unit;
    B42 = sparse(OPS,OPS);
    B43 = (1.0 - BS.gam) / BS.gam * BS.ga * unit;
    B44 = sparse(OPS,OPS) + RAY.nu4 * spdiags(RL,0, OPS, OPS);

    %% Adjust the operator for the coupled BC
    bdex = 1:NZ:OPS;
    GPHI = spdiags(DZTA(1,:)', 0, NX, NX);
    UNIT = spdiags(ones(OPS,1), 0, NX, NX);
    
    % THE BOUNDARY CONDITION MUST BE ON LN(RHO)...
    %B11(bdex,bdex) = (-GPHI);
    %B12(bdex,bdex) = RSBC;
    
    %B21(bdex,bdex) = (-GPHI);
    %B22(bdex,bdex) = UNIT;
    
    B31(bdex,bdex) = (-GPHI);
    B32(bdex,bdex) = UNIT;
    
    %B41(bdex,bdex) = (-GPHI);
    %B42(bdex,bdex) = RSBC;
    
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

    LD = [LD11 LD12 LD13 LD14 ; ...
          LD21 LD22 LD23 LD24 ; ...
          LD31 LD32 LD33 LD34 ; ...
          LD41 LD42 LD43 LD44];

    %% Assemble the force vector
    h_hat = reshape(dzdh .* DZTA ./ dxidz,OPS,1);
    
    F11 = zeros(OPS,1);
    F21 = - h_hat .* BS.ga;
    F31 = zeros(OPS,1);
    F41 = zeros(OPS,1);
    
    %% Adjust the force vector for the coupled BC
    %F11(bdex) = (ujref(1,:) .* DZTA(1,:))';
    %F21(bdex) = (ujref(1,:) .* DZTA(1,:))';
    F31(bdex) = (ujref(1,:) .* DZTA(1,:))';
    %F41(bdex) = (ujref(1,:) .* DZTA(1,:))';
    
    FF = [F11 ; F21 ; F31 ; F41];
end