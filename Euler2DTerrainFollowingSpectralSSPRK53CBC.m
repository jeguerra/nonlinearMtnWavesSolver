% Computes the semi-analytical solution to the steady, linearized Euler equations
% in terrain following coordinates using a coordinate transformation from
% XZ to alpha-eta both from -pi to pi. The vertical boundary condition is
% also tranformed in the vertical so that infinity maps to eta = pi. The
% reference state is a standard atmosphere of piece-wise linear temperature
% profiles with a smooth zonal jet given. Pressure and density initialized
% in hydrostatic balance.

clc
clear
%close all
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 80; % Expansion order matches physical grid
NXO = 80; % Expansion order
NZ = 60; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;

%% Set the test case and global parameters
TestCase = 'ShearJetSchar'; BC = 1;
%TestCase = 'ShearJetScharCBVF'; BC = 1;
%TestCase = 'ClassicalSchar'; BC = 1;
%TestCase = 'AndesMtn'; BC = 1;

z0 = 0.0;
gam = 1.4;
Rd = 287.06;
cp = 1004.5;
cv = cp - Rd;
ga = 9.80616;
p0 = 1.0E5;
if strcmp(TestCase,'ShearJetSchar') == true
    zH = 35000.0;
    l1 = -60000.0;
    l2 = 60000.0;
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
    width = 10000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 100.0;
    mtnh = [int2str(hC) 'm'];
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    zH = 35000.0;
    l1 = -60000.0;
    l2 = 60000.0;
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
    width = 10000.0;
    nu1 = hfactor * 1.0 * 1.0E-2; nu2 = hfactor * 1.0 * 1.0E-2;
    nu3 = hfactor * 1.0 * 1.0E-2; nu4 = hfactor * 1.0 * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ClassicalSchar') == true
    zH = 35000.0;
    l1 = -60000.0;
    l2 = 60000.0;
    L = abs(l2 - l1);
    GAMT = 0.0;
    HT = 0.0;
    GAMS = 0.0;
    HML = 0.0;
    HS = 0.0;
    T0 = 300.0;
    BVF = 0.01;
    depth = 10000.0;
    width = 10000.0;
    hfactor = 1.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    u0 = 10.0;
    uj = 0.0;
    b = 0.0;
elseif strcmp(TestCase,'AndesMtn') == true
    zH = 40000.0;
    l1 = -200000.0;
    l2 = 200000.0;
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
    width = 50000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 1000.0;
    mtnh = [int2str(hC) 'm'];
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

%% Compute the LHS coefficient matrix and force vector for the test case
[LD,FF,REFS] = ...
computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL);

%[LD,FF,REFS] = ...
%computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL);

%[LD,FF,REFS] = ...
%computeCoeffMatrixForceCBC_HerLag(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL);

%% Get the boundary conditions
[FFBC,SOL,sysDex] = GetAdjust4CBC(BC,NX,NZ,OPS,FF);

%% Solve the hyperbolic problem using SSP-RK53
%
disp('Solve with explicit SSP-RK53.');
tic
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = FFBC(sysDex,1);
%A = LD(sysDex,sysDex)' * LD(sysDex,sysDex);
%b = LD(sysDex,sysDex)' * FFBC(sysDex,1);
clear LD FF FFBC;
%
%% Check the eigenvalues of the RHS operator before time integration
[dvecs, dlambda] = eigs(A,20,'lr');
diag(dlambda)
%%
SOL(sysDex) = dvecs(:,1);
uxz = reshape(SOL((1:OPS),1),NZ,NX);
whxz = reshape(SOL((1:OPS) + OPS,1),NZ,NX);
rxz = reshape(SOL((1:OPS) + 2*OPS,1),NZ,NX);
pxz = reshape(SOL((1:OPS) + 3*OPS,1),NZ,NX);

uxz(:,end) = uxz(:,1);
whxz(:,end) = whxz(:,1);
wf = sqrt(REFS.rref0) * REFS.rref.^(-0.5);
wxz = wf .* whxz;
rxz(:,end) = rxz(:,1);
pxz(:,end) = pxz(:,1);

%% Plot the solution in the native grids
%
% NATIVE GRID PLOTS
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); surf(REFS.XL,REFS.ZTL,real(uxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U (m/s)');
subplot(1,2,2); surf(REFS.XL,REFS.ZTL,real(wxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Vertical Velocity W (m/s)');
%
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); surf(REFS.XL,REFS.ZTL,real(rxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Density (kg/m^3)');
subplot(1,2,2); surf(REFS.XL,REFS.ZTL,real(pxz)); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Pressure (Pa)');
drawnow
pause;
%}

% Time step (fraction of a second)
DT = 1.0E-2;
% End time in seconds (HR hours)
HR = 1;
ET = HR * 60 * 60;
%TI = DT:DT:ET;

sol0 = zeros(length(b),1);

%% Explitcit SSP RK53
%
for tt=2:500%length(TI)
    % Initialize the RHS
    RHS = b - A * sol0;
    % Stage 1
    c1 = 0.377268915331368;
    sol1 = sol0 + c1 * DT * RHS;
    RHS = b - A * sol1;
    sol2 = sol1 + c1 * DT * RHS;
    % First linear combination
    LC1_s0 = 0.355909775063327;
    LC1_s2 = 0.644090224936674;
    sol3 = LC1_s0 * sol0 + LC1_s2 * sol2;
    %
    % Stage 2
    c2 = 0.242995220537396;
    RHS = b - A * sol2;
    sol3 = sol3 + c2 * DT * RHS;
    % Second linear combination
    LC2_s0 = 0.367933791638137;
    LC2_s3 = 0.632066208361863;
    sol0 = LC2_s0 * sol0 + LC2_s3 * sol3;
    %
    % Stage 3
    c3 = 0.238458932846290;
    RHS = b - A * sol3;
    sol0 = sol0 + c3 * DT * RHS;
    % Third linear combination
    LC3_s0 = 0.762406163401431;
    LC3_s2 = 0.237593836598569;
    sol5 = LC3_s0 * sol0 + LC3_s2 * sol2;
    %
    % Stage 4
    c4 = 0.287632146308408;
    RHS = b - A * sol0;
    sol5 = sol5 + c4 * DT * RHS;
    %
    % Update the solution
    sol0 = sol5;
    disp(['Time: ' num2str((tt-1)*DT) ' RHS Norm: ' num2str(norm(RHS))]);
end
%}
%% Explicit SSP RK3
%{
for tt=2:500%length(TI)
    % Initialize the RHS
    RHS = b - A * sol0;
    % Stage 1
    c1 = 1.0;
    sol1 = sol0 + c1 * DT * RHS;
    % First linear combination
    LC1_s0 = 0.75;
    LC1_s1 = 0.25;
    sol2 = LC1_s0 * sol0 + LC1_s1 * sol1;
    %
    % Stage 2
    c2 = 0.25;
    RHS = b - A * sol1;
    sol2 = sol2 + c2 * DT * RHS;
    % Second linear combination
    LC2_s0 = 1.0 / 3.0;
    LC2_s2 = 2.0 / 3.0;
    sol3 = LC2_s0 * sol0 + LC2_s2 * sol2;
    %
    % Stage 3
    c3 = 2.0 / 3.0;
    RHS = b - A * sol2;
    sol3 = sol3 + c3 * DT * RHS;
    %
    % Update the solution
    sol0 = sol3;
    disp(['Time: ' num2str((tt-1)*DT) ' RHS Norm: ' num2str(norm(RHS))]);
end
%}

sol = sol0;
clear A b;
toc
%}

%% Get the solution fields
SOL(sysDex) = sol;
clear sol;
%
uxz = reshape(SOL((1:OPS)),NZ,NX);
whxz = reshape(SOL((1:OPS) + OPS),NZ,NX);
rxz = reshape(SOL((1:OPS) + 2*OPS),NZ,NX);
pxz = reshape(SOL((1:OPS) + 3*OPS),NZ,NX);

%% Convert \hat{w} to w using the reference density profile
wf = sqrt(REFS.rref0) * REFS.rref.^(-0.5);
wxz = wf .* whxz;

%% Interpolate to a regular grid using Hermite and Legendre transforms'
%
NXI = 600;
NZI = 300;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(pxz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}
%% Interpolate to a regular grid using Hermite and Laguerre transforms'
%{
DXI = 100.0;
DZI = 50.0;
xint = l1:DXI:l2; NXI = length(xint);
zint = 0.0:DZI:zH; NZI = length(zint);
%[zint, ~] = lagdifJEG(NZI, 1, 2.0 * DS.zH);
[uxzint, XINT, ZINT, ZLINT] = HerTransLagTrans(REFS, DS, real(uxz), NXI, NZI, xint, zint);
[wxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(wxz), NXI, NZI, xint, zint);
[rxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(rxz), NXI, NZI, xint, zint);
[pxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(pxz), NXI, NZI, xint, zint);

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

%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
contourf(XI,ZI,ujref,31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Background Horizontal Velocity U (m/s)');
%}
%{
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref + uxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U $(m~s^{-1})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;

fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,wxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Vertical Velocity W $(m~s^{-1})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
screen2png(['UWREferenceSolution' mtnh '.png']);
%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI, ... 
    (exp(lrref + rxzint) .* (ujref + uxzint) .* wxzint),31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
title('Momentum Flux $(kg~m^{-1}~s^{-2})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['MFluxREferenceSolution' mtnh '.png']);
drawnow
%}
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,rxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Perturbation Log Density $(kg m^{-3})$','FontWeight','normal','Interpreter','latex');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
subplot(1,2,2); contourf(1.0E-3 * XI,1.0E-3 * ZI,pxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Perturbation Log Pressure (Pa)','FontWeight','normal','Interpreter','latex');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,abs(uxzint) ./ abs(ujref),31,'LineStyle','none'); colorbar;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Nonlinearity Ratio $\left| \frac{u}{\bar{u}} \right|$','FontSize',32,'FontWeight','normal','Interpreter','latex');
%}
drawnow
%}
%% Compute the local Ri number and plot ...
%{
dlrho = REFS.dlrref + REFS.sig .* (REFS.DDZ * real(rxz));
duj = REFS.dujref + REFS.sig .* (REFS.DDZ * real(uxz));
Ri = -ga * dlrho ./ (duj.^2);

RiREF = -BS.ga * REFS.dlrref(:,1);
RiREF = RiREF ./ (REFS.dujref(:,1).^2);

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
semilogx(Ri(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',1.5);
semilogx(RiREF,1.0E-3*REFS.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on;
xlabel('Ri','FontSize',30);
%ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
xlim([0.1 1.0E4]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit([mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

dirname = '../ShearJetSchar/';
fname = [dirname 'RI_TEST_' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
[dujint, ~, ~] = HerTransLegInterp(REFS, DS, real(duj), NXI, NZI, 0, 0);
contourf(1.0E-3 * XI,1.0E-3 * ZI,dujint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
title('Vertical Shear $s^{-1}$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['VerticalShear' mtnh '.png']);
drawnow
%}

%% Compute the scaling constants needed for residual diffusion
lrho = REFS.lrref + rxz;
rho = exp(lrho);
lp = REFS.lpref + pxz;
p = exp(lp);
P = exp(REFS.lpref);
R = exp(REFS.lrref);
pt = p ./ (Rd * rho) .* (p0 * p.^(-1)).^(Rd / cp);
PT = p ./ (Rd * R) .* (p0 * P.^(-1)).^(Rd / cp);
RT = (rho .* pt) - (R .* PT);
UINF = norm(uxz - mean(mean(uxz)),Inf);
WINF = norm(wxz - mean(mean(wxz)),Inf);
R = exp(rxz);
RINF = norm(R - mean(mean(R)),Inf);
RTINF = norm(RT - mean(mean(RT)),Inf);
disp('Scaling constants for DynSGS coefficients:');
disp(['|| u - U_bar ||_max = ' num2str(UINF)]);
disp(['\| w - W_bar ||_max = ' num2str(WINF)]);
disp(['\| rho - rho_bar ||_max = ' num2str(RINF)]);
disp(['\| rhoTheta - rhoTheta_bar ||_max = ' num2str(RTINF)]);

%% Compute the local convective stability parameter and plot...
%
dlpres = REFS.dlpref + REFS.sig .* (REFS.DDZ * real(pxz));
rho = exp(lrho);
dlpt = 1 / gam * dlpres - dlrho;
temp = p ./ (Rd * rho);
conv = temp .* dlpt;

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
plot(conv(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
grid on;
xlabel('$\frac{T}{\theta} \frac{d \theta}{d z}$','FontSize',30,'Interpreter','latex');
%ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
%xlim([0.1 1.0E3]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit([mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

dirname = '../ShearJetSchar/';
fname = [dirname 'CONV_TEST_' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%}
%% Debug
%
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(2,2,1); surf(XI,ZI,uxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Total Horizontal Velocity U (m/s)');
subplot(2,2,2); surf(XI,ZI,wxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Vertical Velocity W (m/s)');
subplot(2,2,3); surf(XI,ZI,exp(rxzint)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Perturbation Density (kg/m^3)');
subplot(2,2,4); surf(XI,ZI,exp(pxzint)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Perturbation Pressure (Pa)');
drawnow
%}

%% Plot the terrain forcing
%{
subplot(2,2,1); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS)),NZ,NX)); colorbar; xlim([-15000.0 15000.0]); ylim([0.0 1000.0]);
title('Total Horizontal Velocity U (m/s)');
subplot(2,2,2); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS) + OPS),NZ,NX)); colorbar; ylim([0.0 1000.0]);
title('Vertical Velocity W (m/s)');
subplot(2,2,3); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS) + 2*OPS),NZ,NX)); colorbar; ylim([0.0 1000.0]);
title('Perturbation Density (kg/m^3)');
subplot(2,2,4); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS) + 3*OPS),NZ,NX)); colorbar; ylim([0.0 1000.0]);
title('Perturbation Pressure (Pa)');
drawnow
%}

%% Save the data
%{
close all;
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceHER' char(TestCase) int2str(hC) '.mat'];
save(fileStore);
%}
