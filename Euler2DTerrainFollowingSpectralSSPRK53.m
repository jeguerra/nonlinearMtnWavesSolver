%% COMPUTES TRANSIENT LNP-LNT 2D MOUNTAIN WAVE PROBLEM IN 4 TEST CONFIGURATIONS:

% 1) 'ShearJetSchar' Discontinous background with strong shear
% 2) 'ShearJetScharCBVF' Uniform background with strong shear
% 3) 'ClassicalSchar' The typical Schar mountain test with uniform
% background and constant wind
% 4) 'AndesMtn' Same as 1) but with real input terrain data

clc
clear
close all
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 128; % Expansion order matches physical grid
NZ = 96; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;
iW = 1;
iP = 2;
iT = 3;

%% Set the test case and global parameters
TestCase = 'ShearJetSchar'; BC = 0;
%TestCase = 'ShearJetScharCBVF'; BC = 3;
%TestCase = 'ClassicalSchar'; BC = 3;
%TestCase = 'AndesMtn'; BC = 3;

z0 = 0.0;
gam = 1.4;
Rd = 287.06;
cp = 1004.5;
cv = cp - Rd;
ga = 9.80616;
p0 = 1.0E5;
kappa = Rd / cp;
if strcmp(TestCase,'ShearJetSchar') == true
    zH = 36000.0;
    l1 = -1.0E4 * 3.0 * pi;
    l2 = 1.0E4 * 3.0 * pi;
    L = abs(l2 - l1);
    GAMT = -0.0065;
    HT = 11000.0;
    GAMS = 0.001;
    HML = 9000.0;
    HS = 20000.0;
    T0 = 300.0;
    BVF = 0.0;
    hfactor = 1.0;
    depth = 10000.0;
    width = 20000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    zH = 35000.0;
    l1 = -1.0E4 * 2.0 * pi;
    l2 = 1.0E4 * 2.0 * pi;
    L = abs(l2 - l1);
    GAMT = 0.0;
    HT = 0.0;
    GAMS = 0.0;
    HML = 0.0;
    HS = 0.0;
    T0 = 300.0;
    BVF = 0.01;
    hfactor = 1.0;
    depth = 10000.0;
    width = 15000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ClassicalSchar') == true
    zH = 35000.0;
    l1 = -1.0E4 * 2.0 * pi;
    l2 = 1.0E4 * 2.0 * pi;
    L = abs(l2 - l1);
    GAMT = 0.0;
    HT = 0.0;
    GAMS = 0.0;
    HML = 0.0;
    HS = 0.0;
    T0 = 300.0;
    BVF = 0.01;
    depth = 10000.0;
    width = 15000.0;
    hfactor = 1.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 0.0;
    b = 0.0;
elseif strcmp(TestCase,'AndesMtn') == true
    zH = 40000.0;
    l1 = -1.0E5 * 2.0 * pi;
    l2 = 1.0E5 * 2.0 * pi;
    L = abs(l2 - l1);
    GAMT = -0.0065;
    HT = 11000.0;
    GAMS = 0.001;
    HML = 9000.0;
    HS = 20000.0;
    T0 = 300.0;
    BVF = 0.0;
    hfactor = 1.0;
    depth = 15000.0;
    width = 101000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 100.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '100m';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
end

%% Set up physical parameters for basic state(taken from Tempest defaults)
BS = struct('gam',gam,'Rd',Rd,'cp',cp,'cv',cv,'GAMT',GAMT,'HT',HT,'GAMS', ...
            GAMS,'HML',HML,'HS',HS,'ga',ga,'p0',p0,'T0',T0,'BVF',BVF);

%% Set up the jet and mountain profile parameters
UJ = struct('u0',u0,'uj',uj,'b',b,'ga',ga);
DS = struct('z0',z0,'zH',zH,'l1',l1,'l2',l2,'L',L,'aC',aC,'lC',lC,'hC',hC);

%% Set up the Rayleigh Layer with a coefficient one order of magnitude less than the order of the wave field
RAY = struct('depth',depth,'width',width,'nu1',nu1,'nu2',nu2,'nu3',nu3,'nu4',nu4);

%% Compute the initialization and grid
[REFS, DOPS] = computeGridRefState_LogPLogTh(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL);

%% Get the boundary conditions
[SOL,sysDex] = GetAdjust4CBC(REFS, BC, NX, NZ, OPS);

%% Compute the LHS coefficient matrix and force vector for the test case
%[LD, FF] = ...
%computeCoeffMatrixForce_LogPLogTh(BS, RAY, REFS);
ZSPR = sparse(OPS,OPS);
%{
LD = [DOPS.LD11 DOPS.LD12 DOPS.LD13 ZSPR;      ...
      ZSPR      DOPS.LD22 DOPS.LD23 DOPS.LD24; ...
      DOPS.LD31 DOPS.LD32 DOPS.LD33 ZSPR;      ...
      ZSPR      DOPS.LD42 ZSPR      DOPS.LD44];
%}
LD1 = [DOPS.LD11 ZSPR ZSPR ZSPR;            ...
       ZSPR      DOPS.LD22 ZSPR ZSPR;       ...
       ZSPR ZSPR            DOPS.LD33 ZSPR; ...
       ZSPR ZSPR ZSPR                 DOPS.LD44];
  
LD2 = [ZSPR      DOPS.LD12 DOPS.LD13 ZSPR;      ...
       ZSPR      ZSPR      DOPS.LD23 DOPS.LD24; ...
       DOPS.LD31 DOPS.LD32 ZSPR      ZSPR;      ...
       ZSPR      DOPS.LD42 ZSPR      ZSPR]; 

%% Compute coupled multipoint BC by adjusting columns of LD
ubdex = 1:NZ:(OPS - NZ + 1);
wbdex = ubdex + iW*OPS;
dhdx = spdiags((REFS.DZT(1,:))', 0, NX, NX);
% Apply column adjustment for the multipoint coupled BC
LD1(:,ubdex) = LD1(:,ubdex) + LD1(:,wbdex) * dhdx;
LD2(:,ubdex) = LD2(:,ubdex) + LD2(:,wbdex) * dhdx;
% Compute RHS scaling
WBC = REFS.DZT(1,:) .* REFS.ujref(1,:);

%% Apply boundary forcing and set up the system
AN1 = LD1(sysDex,sysDex);
AN2 = LD2(sysDex,sysDex);
b = -(LD1(:,wbdex) * WBC') - (LD2(:,wbdex) * WBC'); clear LD1 LD2;
bN = b(sysDex); clear b;

%% Solve the hyperbolic problem using SSP-RK53
%{
disp('Solve with explicit SSP-RK53.');
tic
spparms('spumoni',2);
A = LD;
b = FF;
AN = A;
bN = b;
toc; disp('Compute coefficient matrix... DONE.');
clear A b LD FF;
%{
%% Check the eigenvalues of the RHS operator before time integration
[dvecs, dlambda] = eigs(AN(sysDex,sysDex),10,'lr');
diag(dlambda)
%%
SOL(sysDex) = dvecs(:,1);
uxz = reshape(SOL((1:OPS),1),NZ,NX);
rxz = reshape(SOL((1:OPS) + OPS,1),NZ,NX);
wxz = reshape(SOL((1:OPS) + 2*OPS,1),NZ,NX);
pxz = reshape(SOL((1:OPS) + 3*OPS,1),NZ,NX);

%% Plot the solution in the native grids
%
% NATIVE GRID PLOTS
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); surf(REFS.XL,REFS.ZTL,real(uxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('$u$ $(ms^{-1})$');
subplot(1,2,2); surf(REFS.XL,REFS.ZTL,real(wxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('$w$ $(ms^{-1})$');
%
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); surf(REFS.XL,REFS.ZTL,real(rxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation $\ln p$ $(Pa)$');
subplot(1,2,2); surf(REFS.XL,REFS.ZTL,real(pxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation $\ln \theta$ $(K)$');
drawnow
pause;
%}
%}
% Time step (fraction of a second)
DT = 0.06;
% End time in seconds (HR hours)
HR = 5.0;
ET = HR * 60 * 60;
TI = DT:DT:ET;
% Output times as an integer multiple of DT
OTI = 50;

%% Set storage for solution vectors and initialize
sol = zeros(length(SOL),3);
for ss=1:2
    sol(:,ss) = SOL;
end

%% Index arrays for prognostic components
udex = 1:OPS;
wdex = iW*OPS+1:iP*OPS;
pdex = iP*OPS+1:iT*OPS;
tdex = iT*OPS+1:numVar*OPS;

%% Explicit SSP RK93 in low storage form
tic
c1 = 1.0 / 6.0;
c2 = 1.0 / 5.0;
% Initialize the RHS
RHS = bN - AN1 * sol(sysDex,1) - AN2 * sol(sysDex,1);
for tt=1:length(TI)
    if mod(tt,10) == 0
        % Compute the nonlinear operator for residual diffusion (EXPENSIVE)
        sol(sysDex,3) = RHS;
        RVD = computeResidualViscOperator_LogPLogTh(REFS, sol(:,3));
        % Apply to potential temperature only
        rvt = RVD.RVD44 * sol(tdex,1);
        % Put viscosity tendency in auxiliary storage
        sol(sysDex,3) = 0.0 * sol(sysDex,3);
        sol(tdex,3) = rvt; clear rvt;
    else
        sol(sysDex,3) = 0.0 * sol(sysDex,3);
    end
    % First stage
    sol(sysDex,1) = sol(sysDex,1) +  c1 * DT * (RHS + sol(sysDex,3));
    % Copy to the second storage
    sol(sysDex,2) = sol(sysDex,1);
    % Compute stages 2 to 5
    for ii=2:5
        sol(sysDex,1) = sol(sysDex,1) + c1 * DT * (RHS + sol(sysDex,3));
        % Update the RHS
        RHS = bN - AN1 * sol(sysDex,1) - AN2 * sol(sysDex,1);
        %{
        spmd(2)
            if labindex == 1
                rhs = AN1 * sol(sysDex,1);
            else
                rhs = AN2 * sol(sysDex,1);
            end
        end
        RHS = bN - rhs{1} - rhs{2};
        %}
    end
    % Compute stage 6
    sol(sysDex,1) = 3.0 * sol(sysDex,2) + 2.0 * ...
        (sol(sysDex,1) +  c1 * DT * (RHS + sol(sysDex,3)));
    sol(sysDex,1) = c2 * sol(sysDex,1);
    % Compute stages 7 to 9
    for ii=7:9
        sol(sysDex,1) = sol(sysDex,1) + c1 * DT * (RHS + sol(sysDex,3));
        % Update the RHS
        RHS = bN - AN1 * sol(sysDex,1) - AN2 * sol(sysDex,1);
        %{
        spmd(2)
            if labindex == 1
                rhs = AN1 * sol(sysDex,1);
            else
                rhs = AN2 * sol(sysDex,1);
            end
        end
        RHS = bN - rhs{1} - rhs{2};
        %}
    end
    
    % Update the solution 
    if mod(tt,OTI) == 0
        disp(['Time: ' num2str((tt-1)*DT) ' RHS Norm: ' num2str(norm(RHS))]);
    end
    
    % Zero out auxiliary storage
    sol(sysDex,3) = 0.0 * sol(sysDex,3);
end
%{
%% Set storage for solution vectors and initialize
sol = zeros(length(SOL),5);
for ss=1:5
    sol(:,ss) = SOL;
end     
    
%% Explitcit SSP RK53 with full storage
tic
%matMul = @(xVec) computeCoeffMatrixMulLogPLogTh(REFS, DOPS, xVec, 1:numVar*OPS);
for tt=1:length(TI)
    % Initialize the RHS
    RHS = bN - AN * sol(sysDex,1);
    %RHS = bN - matMul(sol(:,1));
    % Stage 1
    c1 = 0.377268915331368;
    sol(sysDex,2) = sol(sysDex,1) + c1 * DT * RHS;
    RHS = bN - AN * sol(sysDex,2);
    %RHS = bN - matMul(sol(:,2));
    sol(sysDex,3) = sol(sysDex,2) + c1 * DT * RHS;
    % First linear combination
    LC1_s0 = 0.355909775063327;
    LC1_s2 = 0.644090224936674;
    sol(sysDex,4) = LC1_s0 * sol(sysDex,1) + LC1_s2 * sol(sysDex,3);
    %
    % Stage 2
    c2 = 0.242995220537396;
    RHS = bN - AN * sol(sysDex,3);
    %RHS = bN - matMul(sol(:,3));
    sol(sysDex,4) = sol(sysDex,4) + c2 * DT * RHS;
    % Second linear combination
    LC2_s0 = 0.367933791638137;
    LC2_s3 = 0.632066208361863;
    sol(sysDex,1) = LC2_s0 * sol(sysDex,1) + LC2_s3 * sol(sysDex,4);
    %
    % Stage 3
    c3 = 0.238458932846290;
    RHS = bN - AN * sol(sysDex,4);
    %RHS = bN - matMul(sol(:,4));
    sol(sysDex,1) = sol(sysDex,1) + c3 * DT * RHS;
    % Third linear combination
    LC3_s0 = 0.762406163401431;
    LC3_s2 = 0.237593836598569;
    sol(sysDex,5) = LC3_s0 * sol(sysDex,1) + LC3_s2 * sol(sysDex,3);
    %
    % Stage 4
    c4 = 0.287632146308408;
    RHS = bN - AN * sol(sysDex,1);
    %RHS = bN - matMul(sol(:,1));
    sol(sysDex,5) = sol(sysDex,5) + c4 * DT * RHS;
    %
    % Update the solution (currently NOT storing history...)
    sol(sysDex,1) = sol(sysDex,5);
    
    if mod(tt,OTI) == 0
        RHS = bN - AN * sol(sysDex,1);
        disp(['Time: ' num2str((tt-1)*DT) ' RHS Norm: ' num2str(norm(RHS))]);
    end
end
toc
%}

%% Get the solution fields
SOL(sysDex) = sol(sysDex,1);
clear sol;
%
uxz = reshape(SOL((1:OPS)),NZ,NX);
wxz = reshape(SOL((1:OPS) + OPS),NZ,NX);
rxz = reshape(SOL((1:OPS) + 2*OPS),NZ,NX);
pxz = reshape(SOL((1:OPS) + 3*OPS),NZ,NX);

%% Interpolate to a regular grid using Hermite and Laguerre transforms'
%
NXI = 3000;
NZI = 300;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(pxz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}
%% Plot the solution in the native grids
%{
% NATIVE GRID PLOTS
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,real(uxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U (m/s)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(wxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Vertical Velocity W (m/s)');
%
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,real(rxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Density (kg/m^3)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(pxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Pressure (Pa)');
drawnow
%}

% Use the NCL hotcold colormap
cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);

% INTERPOLATED GRID PLOTS
%
width = 0.0;
depth = 0.0;
%% Compute the reference state initialization
%
if strcmp(TestCase,'ShearJetSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZLINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [lprefU,~,dlprefU,~] = computeBackgroundPressureCBVF(BS, ZLINT);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
elseif strcmp(TestCase,'ClassicalSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [ujref, ~] = computeJetProfileUniform(UJ, lpref);
elseif strcmp(TestCase,'AndesMtn') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZLINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
end
%}
dlthref = 1.0 / BS.gam * dlpref - dlrref;

%%
figure;
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,uxzint,31); colorbar; grid on; cm = caxis;
%contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-200 300]);
%ylim([0 15]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('\textsf{$U^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['UREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,wxzint,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-100 100]);
%ylim([0 15]);
title('\textsf{$W^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['WREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,rxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln p)^{\prime} ~~ (Pa)$');
subplot(1,2,2); contourf(1.0E-3 * XI,1.0E-3 * ZI,pxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln \theta)^{\prime} ~~ (K)$');
drawnow
%}

%% Debug plots to check the boundary up close
%{
figure;
subplot(2,2,1); surf(XI,ZI,uxzint); colorbar; xlim([-25000.0 25000.0]); ylim([0.0 2000.0]);
title('U (m/s)'); %zlim([-0.1 0.1]);
subplot(2,2,2); surf(XI,ZI,wxzint); colorbar; xlim([-25000.0 25000.0]); ylim([0.0 2000.0]);
title('W (m/s)');
subplot(2,2,3); surf(XI,ZI,exp(lrref) .* (exp(rxzint) - 1.0)); colorbar; xlim([-30000.0 30000.0]); ylim([0.0 2000.0]);
title('$(\ln p)^{\prime}$ (Pa)');
subplot(2,2,4); surf(XI,ZI,exp(lpref) .* (exp(pxzint) - 1.0)); colorbar; xlim([-30000.0 30000.0]); ylim([0.0 2000.0]);
title('$(\ln \theta)^{\prime}$ (K)');
drawnow
%}

%% Save the data
%{
close all;
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceHER' char(TestCase) int2str(hC) '.mat'];
save(fileStore);
%}
