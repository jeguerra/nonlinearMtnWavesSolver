function [LDA,FFA,REFS] = computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL)
    %% Compute the Hermite and Legendre points and derivatives for this grid
    % Set the boundary indices and operator dimension
    OPS = NX*NZ;
    tdex = NZ:NZ:OPS;
    bdex = 1:NZ:(OPS - NZ + 1);
    rdex = (OPS - NZ + 2):OPS - 1;
    NO = length(bdex);
    NR = length(rdex);
    
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
    %DDX_H(:,1) = DDX_H(:,1) + DDX_H(:,end);
    %}
    [zlc, ~] = chebdif(NZ, 1);
    zl = DS.zH * 0.5 * (zlc + 1.0);
    zlc = 0.5 * (zlc + 1.0);
    DDZ_L = (1.0 / DS.zH) * poldif(zlc, 1);
    %}
          
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
    % Compute the horizontal metric derivative
    dadx = (1.0 + DZT.^2).^(0.5);
    dxidx = sigma .* DZT;
    
    %% Compute the Rayleigh field
    [rayField, ~] = computeRayleighXZ(DS,1.0,RAY.depth,RAY.width,XL,ZL,applyTopRL,applyLateralRL);
    %[rayField, ~] = computeRayleighPolar(DS,1.0,RAY.depth,XL,ZL);
    %[rayField, ~] = computeRayleighEllipse(DS,1.0,RAY.depth,RAY.width,XL,ZL);
    RL = reshape(rayField,OPS,1);

    %% Compute the reference state initialization
    if strcmp(TestCase,'ShearJetSchar') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(BS, DS.zH, zl, ZTL, RAY);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
        %[lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, DS.zH, zl, ZL, RAY);
        %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
    elseif strcmp(TestCase,'ShearJetScharCBVF') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressureCBVF(BS, ZTL);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
        %[lprefU,~,dlprefU,~] = computeBackgroundPressureCBVF(BS, ZL);
        %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
    elseif strcmp(TestCase,'ClassicalSchar') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressureCBVF(BS, ZTL);
        [ujref,dujref] = computeJetProfileUniform(UJ, lpref);
    elseif strcmp(TestCase,'AndesMtn') == true
        [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(BS, DS.zH, zl, ZTL, RAY);
        [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
        %[lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, DS.zH, zl, ZL, RAY);
        %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
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
    
    fig = figure('Position',[0 0 1200 1200]); fig.Color = 'w';
    Im = dlthref(:,1) + 0.5 * dujref(:,1) ./ ujref(:,1) - 0.5 * dlpref(:,1);
    MZ = exp(sigma(:,1) .* Im);
    for xx=2:length(MZ)
        MZ(xx) = MZ(xx) * MZ(xx-1);
    end
    plot(MZ,1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Amplitude Gain','FontSize',30);
    ylabel('Altitude (km)','FontSize',30);
    ylim([0.0 25.0]);
    %xlim([1.0 1.0002]);
    fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
    drawnow;
    dirname = '../ShearJetSchar/';
    fname = [dirname 'VERTICAL_WAVENUMBER'];
    screen2png(fname);
    pause
%}
    
    REFS = struct('ujref',ujref,'dujref',dujref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'dlthref',dlthref,...
        'pref',pref,'rref',rref,'thref',thref,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ',DDZ_L, ...
        'DDX_H',DDX_H,'sigma',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);

    %% Unwrap the derivative matrices into operators onto a state 1D vector
    % Compute the vertical derivatives operator (Legendre expansion)
    DDXI_OP = zeros(OPS);
    for cc=1:NX
        ddex = (1:NZ) + (cc - 1) * NZ;
        DDXI_OP(ddex,ddex) = DDZ_L;
    end
    DDXI_OP = sparse(DDXI_OP);

    % Compute the horizontal derivatives operator (Hermite expansion)
    DDA_OP = zeros(OPS);
    for rr=1:NZ
        ddex = (1:NZ:OPS) + (rr - 1);
        DDA_OP(ddex,ddex) = DDX_H;
    end
    DDA_OP = sparse(DDA_OP);

    %% Assemble the block global operator L
    SIGMA = spdiags(reshape(sigma,OPS,1), 0, OPS, OPS);
    DADX = spdiags(reshape(dadx,OPS,1), 0, OPS, OPS);
    DXIDX = spdiags(reshape(dxidx,OPS,1), 0, OPS, OPS);
    U0 = spdiags(reshape(ujref,OPS,1), 0, OPS, OPS);
    DUDZ = spdiags(reshape(dujref,OPS,1), 0, OPS, OPS);
    DLPDZ = spdiags(reshape(dlpref,OPS,1), 0, OPS, OPS);
    DLRDZ = spdiags(reshape(dlrref,OPS,1), 0, OPS, OPS);
    DLPTDZ = (1.0 / BS.gam * DLPDZ - DLRDZ);
    POR = spdiags(reshape(pref ./ rref,OPS,1), 0,  OPS, OPS);
    U0DX = U0 * DDA_OP;
    DX = DDA_OP;
    %DX = DDA_OP - DXIDX * DDXI_OP;
    %U0DX = U0 * (DDA_OP - DXIDX * DDXI_OP);
    unit = spdiags(ones(OPS,1),0, OPS, OPS);

    % Horizontal momentum LHS
    L11 = U0DX;
    L12 = sparse(OPS,OPS);
    L13 = POR * DX;
    L14 = sparse(OPS,OPS);
    % Vertical momentum LHS
    L21 = sparse(OPS,OPS);
    L22 = U0DX;
    L23 = POR * SIGMA * DDXI_OP;
    L24 = sparse(OPS,OPS);
    % Continuity (log pressure) LHS
    L31 = BS.gam * DDA_OP;
    L32 = BS.gam * SIGMA * DDXI_OP;
    L33 = U0DX;
    L34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    L41 = sparse(OPS,OPS);
    L42 = sparse(OPS,OPS);
    L43 = sparse(OPS,OPS);
    L44 = U0DX;

    %% Assemble the algebraic part (Rayleigh layer on the diagonal)
    % Horizontal momentum LHS
    B11 = sparse(OPS,OPS) + RAY.nu1 * spdiags(RL,0, OPS, OPS);
    B12 = DUDZ;
    B13 = sparse(OPS,OPS);
    B14 = sparse(OPS,OPS);
    % Vertical momentum LHS
    B21 = sparse(OPS,OPS);
    B22 = sparse(OPS,OPS) + RAY.nu2 * spdiags(RL,0, OPS, OPS);
    B23 = BS.ga * (1.0 / BS.gam - 1.0) * unit;
    B24 = - BS.ga * unit;
    % Continuity (log pressure) LHS
    B31 = sparse(OPS,OPS);
    B32 = DLPDZ;
    B33 = sparse(OPS,OPS) + RAY.nu3 * spdiags(RL,0, OPS, OPS);
    B34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    B41 = sparse(OPS,OPS);
    B42 = DLPTDZ;
    B43 = sparse(OPS,OPS);
    B44 = sparse(OPS,OPS) + RAY.nu4 * spdiags(RL,0, OPS, OPS);
    
    %% Constraint the top and bottom boundaries
    %{
    % No horizontal transport of vertical velocity
    B12(tdex,tdex) = zeros(NO);
    % No vertical momentum constraint along boundary
    L22(tdex,tdex) = zeros(NO);
    L23(tdex,tdex) = zeros(NO);
    B23(tdex,tdex) = zeros(NO);
    B24(tdex,tdex) = zeros(NO);
    % No vertical transport of pressure 
    B32(tdex,tdex) = zeros(NO);
    % No vertical transport of entropy 
    B42(tdex,tdex) = zeros(NO);
    %
    %L11(bdex,bdex) = zeros(NO);
    %L22(bdex,bdex) = zeros(NO);
    %LD22(bdex,bdex) = zeros(NO);
    %LD23(bdex,bdex) = zeros(NO);
    %LD24(bdex,bdex) = zeros(NO);
    %LD32(bdex,bdex) = zeros(NO);
    %LD42(bdex,bdex) = zeros(NO);
    %}
    
    %% Neumann condition at the right lateral boundary
    %{
    L11(rdex,rdex) = zeros(NR);
    L13(rdex,rdex) = zeros(NR);
    L22(rdex,rdex) = zeros(NR);
    L31(rdex,rdex) = zeros(NR);
    L33(rdex,rdex) = zeros(NR);
    L44(rdex,rdex) = zeros(NR);
    %}
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
    
    %% Assemble the LHS operator
    LD = [LD11 LD12 LD13 LD14 ; ...
          LD21 LD22 LD23 LD24 ; ...
          LD31 LD32 LD33 LD34 ; ...
          LD41 LD42 LD43 LD44];
      
    %% Assemble the force vector
    F11 = zeros(OPS,1);
    F21 = zeros(OPS,1);
    F31 = zeros(OPS,1);
    F41 = zeros(OPS,1);
    FF = [F11 ; F21 ; F31 ; F41];
    
    %% Compute the bottom boundary constraint
    HBC = sparse(NX,4 * OPS);
    UNIT = spdiags(ones(NX,1), 0, NX, NX);
    HX = spdiags((DZT(1,:))', 0, NX, NX);
    HBC(:,bdex) = -HX;
    HBC(:,bdex + OPS) = UNIT;
    
    %% Compute the top boundary constraint
    TBC = sparse(NX,4 * OPS);
    UNIT = spdiags(ones(NX,1), 0, NX, NX);
    TBC(:,tdex + OPS) = UNIT;
    
    %% Augment the system with the Lagrange Multiplier Constraints
    RPAD = zeros(2*NX);
    LDA = [LD HBC' TBC'];
    RAG = [HBC ; TBC];
    RAG = [RAG RPAD];
    LDA = [LDA ; RAG];
    FFA = [FF ; (ujref(1,:) .* DZT(1,:))' ; zeros(NX,1)];
    
    %{
    %% Compute the lateral boundary constraint
    RBC = sparse(NZ-2,4*OPS);
    UNIT = spdiags(ones(NZ-2,1), 0, NZ-2, NZ-2);
    RBC(:,rdex + OPS) = UNIT;
    %
    %% Augment the system with the Lagrange Multiplier Constraints
    RPAD = zeros(2*NX + (NZ - 2));
    LDA = [LD HBC' TBC' RBC'];
    RAG = [HBC ; TBC ; RBC];
    RAG = [RAG RPAD];
    LDA = [LDA ; RAG];
    FFA = [FF ; (ujref(1,:) .* DZT(1,:))' ; zeros(NX,1) ; zeros(NZ-2,1)];
    %}
end