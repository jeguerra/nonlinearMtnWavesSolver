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
NX = 200;
NXO = 60; % Expansion order
NZ = 80; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;

%% Set the test case and global parameters
TestCase = 'AndesMtn'; BC = 0;

z0 = 0.0;
gam = 1.4;
Rd = 287.06;
cp = 1004.5;
cv = cp - Rd;
ga = 9.80616;
p0 = 1.0E5;
if strcmp(TestCase,'AndesMtn') == true
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
    width = 51000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 1000.0;
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
DS = struct('z0',z0,'zH',zH,'l1',l1,'l2',l2,'L',L,'aC',aC,'lC',lC,'hC',hC,'hfilt',hfilt);

%% Set up the Rayleigh Layer with a coefficient one order of magnitude less than the order of the wave field
RAY = struct('depth',depth,'width',width,'nu1',nu1,'nu2',nu2,'nu3',nu3,'nu4',nu4);

%% Compute the LHS coefficient matrix and force vector for the test case
[LD,FF,REFS] = ...
computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL);

%% Get the boundary conditions
[FFBC,SOL,sysDex] = GetAdjust4CBC(BC,NX,NZ,OPS,FF);

%% Solve the system using the matlab linear solver
%
disp('Solve the raw system with matlab default \.');
tic
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = FFBC(sysDex,1);
%spy(A); pause;
%A = LD(sysDex,sysDex)' * LD(sysDex,sysDex);
%b = LD(sysDex,sysDex)' * FFBC(sysDex,1);
clear LD FF FFBC;
sol = A \ b;
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
DXI = 1000.0;
DZI = 250.0;
xint = l1:DXI:l2; NXI = length(xint);
zint = 0.0:DZI:zH; NZI = length(zint);
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, RAY, real(uxz), NXI, NZI, xint', zint');
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(wxz), NXI, NZI, xint', zint');
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(rxz), NXI, NZI, xint', zint');
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(pxz), NXI, NZI, xint', zint');

XI = l2 * XINT;
ZI = ZINT;

%% Plot the solution in the native and interpolated grids
%{
% NATIVE GRID PLOTS
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,real(REFS.ujref + uxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U (m/s)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(wxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Vertical Velocity W (m/s)');

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
width = 0.0;
depth = 0.0;
%% Compute the reference state initialization
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
dlthref = 1.0 / BS.gam * dlpref - dlrref;

%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
contourf(XI,ZI,ujref,31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Background Horizontal Velocity U (m/s)');
%}
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref + uxzint,31); colorbar; grid on;
%subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,uxzint,31); colorbar; grid on;
%contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref,21,'LineWidth',1.0); %grid on; colorbar;
%hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
%title('Total Horizontal Velocity U $(m~s^{-1})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
screen2png(['UREferenceSolution' mtnh '.png']);

fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,wxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Vertical Velocity W $(m~s^{-1})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
screen2png(['WREferenceSolution' mtnh '.png']);
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

%% Compute the local Ri number and plot ...
%
lrho = REFS.lrref + real(rxz);
dlrho = REFS.sigma .* (REFS.DDZ * lrho);
uj = REFS.ujref + real(uxz);
duj = REFS.sigma .* (REFS.DDZ * uj);
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
ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
xlim([0.1 1.0E4]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit('Local Ri Number','FontSize',30,'FontWeight','normal','Interpreter','tex');

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
rho = exp(lrho);
lp = REFS.lpref + real(pxz);
p = exp(lp);
pt = p ./ (Rd * rho) .* (p0 * p.^(-1)).^(Rd / cp);
dpt = REFS.sigma .* (REFS.DDZ * pt);
temp = p ./ (Rd * rho);
conv = (pt ./ temp) .* dpt;

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
plot(conv(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
grid on;
xlabel('$\frac{T}{\theta} \frac{d \theta}{d z}$','FontSize',30,'Interpreter','latex');
ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
%xlim([0.1 1.0E3]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit('Convective Stability','FontSize',30,'FontWeight','normal','Interpreter','tex');

dirname = '../ShearJetSchar/';
fname = [dirname 'CONV_TEST_' num2str(hC)];
drawnow;
screen2png(fname);

%% Debug
%{
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
