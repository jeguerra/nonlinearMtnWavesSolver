function [LD,FF,REFS] = computeCoeffMatrixForceCBC_RhoTheta(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL)
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
    DDX_H(:,1) = DDX_H(:,1) + DDX_H(:,end);
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
        DZT(rr,:) = fxi(rr,:) .* dhdx';
    end
    % Compute the horizontal metric derivative
    dAdX = (1.0 + DZT.^2).^(0.5);
    
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
    rref0 = max(max(rref));
    % Background potential temperature profile
    dlthref = 1.0 / BS.gam * dlpref - dlrref;
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
    ylabel('Elevation (km)');
    ylim([0.0 35.0]);
    drawnow;
    
    subplot(2,2,2);
    plot(pref(:,1) ./ (rref(:,1) * BS.Rd),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    %title('Temperature Profile','FontSize',30);
    xlabel('Temperature (K)');
    ylabel('Elevation (km)');
    ylim([0.0 35.0]);
    drawnow;
    
    subplot(2,2,3);
    plot(0.01*pref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Pressure (hPa)');
    ylabel('Elevation (km)');
    ylim([0.0 35.0]);
    drawnow;
    
    subplot(2,2,4);
    plot(thref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('P. Temperature (K)','FontSize',30);
    ylabel('Elevation (km)');
    ylim([0.0 35.0]);
    drawnow;
    dirname = '../ShearJetSchar/';
    fname = [dirname 'BACKGROUND_PROFILES'];
    screen2png(fname);
    
    figure;
    plot(dujref(:,1),1.0E-3*zl,'k-s','LineWidth',1.5); grid on;
    xlabel('Shear $(s^{-1})$');
    ylabel('Elevation (km)');
    ylim([0.0 35.0]);
    drawnow;
    pause
%}
    
    REFS = struct('ujref',ujref,'dujref',dujref, ...
        'lpref',lpref,'dlpref',dlpref,'lrref',lrref,'dlrref',dlrref,'lthref',lthref,'dlthref',dlthref,...
        'pref',pref,'rref',rref,'thref',thref,'XL',XL,'xi',xi,'ZTL',ZTL,'DZT',DZT,'DDZ',DDZ_L, ...
        'DDX_H',DDX_H,'sigma',sigma,'NX',NX,'NZ',NZ,'TestCase',TestCase,'rref0',rref0,'thref0',thref0);

    %% Unwrap the derivative matrices into operators onto a state 1D vector
    % Compute the vertical derivatives operator (Legendre expansion)
    DDXI_OP = spalloc(OPS, OPS, NZ^2);
    for cc=1:NX
        ddex = (1:NZ) + (cc - 1) * NZ;
        DDXI_OP(ddex,ddex) = DDZ_L;
    end
    %DDXI_OP = sparse(DDXI_OP);

    % Compute the horizontal derivatives operator (Hermite expansion)
    DDA_OP = spalloc(OPS, OPS, NX^2);
    for rr=1:NZ
        ddex = (1:NZ:OPS) + (rr - 1);
        DDA_OP(ddex,ddex) = DDX_H;
    end
    %DDA_OP = sparse(DDA_OP);

    %% Assemble the block global operator L
    OPS = NX*NZ;
    SIGMA = spdiags(reshape(sigma,OPS,1), 0, OPS, OPS);
    DADX = spdiags(reshape(dAdX,OPS,1), 0, OPS, OPS);
    U0 = spdiags(reshape(ujref,OPS,1), 0, OPS, OPS);
    DUDZ = spdiags(reshape(dujref,OPS,1), 0, OPS, OPS);
    DTHDZ = spdiags(reshape(dTref,OPS,1), 0, OPS, OPS);
    DRDZ = spdiags(reshape(rref .* dlrref,OPS,1), 0, OPS, OPS);
    DLRTDZ = spdiags(reshape(dlrref + dlthref,OPS,1), 0, OPS, OPS);
    RHOZ = spdiags(reshape(rref,OPS,1), 0, OPS, OPS);
    THTZ = spdiags(reshape(thref,OPS,1), 0, OPS, OPS);
    RDTZ = spdiags(reshape(pref ./ rref,OPS,1), 0, OPS, OPS);
    IRHOZ = spdiags(reshape(rref.^(-1),OPS,1), 0, OPS, OPS);
    IRTHZ = spdiags(reshape((rref .* thref).^(-1),OPS,1), 0, OPS, OPS);
    PGFT = RDTZ * IRTHZ;
    U0DA = U0 * DDAL_OP;
    
    %unit = spdiags(ones(OPS,1),0, OPS, OPS);

    % Horizontal momentum LHS
    L11 = U0DA;
    L12 = sparse(OPS,OPS);
    L13 = sparse(OPS,OPS);
    L14 = BS.gam * PGFT * DDAL_OP;
    % Vertical momentum LHS
    L21 = sparse(OPS,OPS);
    L22 = U0DA;
    L23 = sparse(OPS,OPS);
    L24 = BS.gam * PGFT * SIGMA * DDXI_OP;
    % Continuity LHS
    L31 = RHOZ * DDAL_OP;
    L32 = RHOZ * SIGMA * DDXI_OP;
    L33 = U0DA;
    L34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    L41 = sparse(OPS,OPS);
    L42 = sparse(OPS,OPS);
    L43 = -THTZ * U0DA;
    L44 = U0DA;

    %% Assemble the algebraic part (Rayleigh layer on the diagonal)
    % Horizontal momentum LHS
    B11 = sparse(OPS,OPS) + RAY.nu1 * spdiags(RL,0, OPS, OPS);
    B12 = DUDZ;
    B13 = sparse(OPS,OPS);
    B14 = sparse(OPS,OPS);
    % Vertical momentum LHS
    B21 = sparse(OPS,OPS);
    B22 = sparse(OPS,OPS) + RAY.nu2 * spdiags(RL,0, OPS, OPS);
    B23 = BS.ga * IRHOZ;
    B24 = -(BS.gam * PGFT * DLRTDZ + BS.gam * BS.ga * IRTHZ);
    % Continuity LHS (using density weighted change of variable in W)
    B31 = sparse(OPS,OPS);
    B32 = DRDZ;
    B33 = sparse(OPS,OPS) + RAY.nu3 * spdiags(RL,0, OPS, OPS);
    B34 = sparse(OPS,OPS);
    % Thermodynamic LHS
    B41 = sparse(OPS,OPS);
    B42 = RHOZ * DTHDZ;
    B43 = sparse(OPS,OPS) - RAY.nu4 * spdiags(RL,0, OPS, OPS) * THTZ;
    B44 = sparse(OPS,OPS) + RAY.nu4 * spdiags(RL,0, OPS, OPS);
    
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
    F11 = zeros(OPS,1);
    F21 = zeros(OPS,1);
    F31 = zeros(OPS,1);
    F41 = zeros(OPS,1);
    FF = [F11 ; F21 ; F31 ; F41];
    
    %% Compute the bottom boundary constraint
    HBC = sparse(NB,4 * OPS);
    HBCT = HBC;
    UNIT = spdiags(ones(NB,1), 0, NB, NB);
    % Row augmentation
    HBC(:,bdex + OPS) = UNIT;
    % Column augmentation
    HBCT(:,bdex + OPS) = UNIT;
    
    %% Compute the top boundary constraint
    TBC = sparse(NT,4 * OPS);
    TBCT = TBC;
    UNIT = spdiags(ones(NT,1), 0, NT, NT);
    % Row augmentation
    TBC(:,tdex + OPS) = UNIT;
    % Column augmentation
    TBCT(:,tdex + OPS) = UNIT;
    
    %% Augment the system with the Lagrange Multiplier Constraints
    RPAD = zeros(NB + NT);
    LDA = [LD HBCT' TBCT'];
    RAG = [HBC ; TBC];
    RAG = [RAG RPAD];
    LDA = [LDA ; RAG];
    FFA = [FF ; (ujref(1,1:NX) .* DZT(1,1:NX))' ; zeros(NT,1)];
end