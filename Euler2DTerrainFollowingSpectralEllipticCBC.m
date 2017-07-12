% Computes the semi-analytical solution to the steady, linearized Euler equations
% in terrain following coordinates using a coordinate transformation from
% XZ to alpha-eta both from -pi to pi. The vertical boundary condition is
% also tranformed in the vertical so that infinity maps to eta = pi. The
% reference state is a standard atmosphere of piece-wise linear temperature
% profiles with a smooth zonal jet given. Pressure and density initialized
% in hydrostatic balance.

clc
clear
close all
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 100;
NZ = 80;
OPS = NX * NZ;
numVar = 4;

%% Set the test case
%TestCase = 'ShearJetSchar'; BC = 1;
%TestCase = 'NonhydroMtn'; BC = 1;
TestCase = 'ClassicalSchar'; BC = 1;

z0 = 0.0;
nu = 1.0;
hfactor = 1.0;
gam = 1.4;
Rd = 287.0;
cp = 1004.5;
cv = cp - Rd;
ga = 9.80616;
p0 = 1.0E5;
if strcmp(TestCase,'ShearJetSchar') == true
    zH = 43000.0;
    l1 = -60000.0;
    l2 = 60000.0;
    L = abs(l2 - l1);
    GAMT = -0.0065;
    HT = 11000.0;
    GAMS = 0.001;
    HML = 9000.0;
    HS = 20000.0;
    T0 = 292.15;
    BVF = 0.0;
    depth = 8000.0;
    width = 8000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 250.0;
    mtnh = '250m';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
    % Jet parameters (see notes)
    %uj1 = 204.0;
    %uj2 = 5.0;
elseif strcmp(TestCase,'NonhydroMtn') == true
    zH = 25000.0;
    l1 = -50000.0;
    l2 = 50000.0;
    L = abs(l2 - l1);
    GAMT = 0.0;
    HT = 0.0;
    GAMS = 0.0;
    HML = 0.0;
    HS = 0.0;
    T0 = 280.0;
    BVF = 0.01;
    depth = 10000.0;
    width = 10000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    aC = 1000.0;
    lC = 0.0;
    mtnh = '1m';
    hC = 1.0;
    u0 = 10.0;
    uj = 0.0;
    b = 0.0;
elseif strcmp(TestCase,'ClassicalSchar') == true
    zH = 28000.0;
    l1 = -33000.0;
    l2 = 33000.0;
    L = abs(l2 - l1);
    GAMT = 0.0;
    HT = 0.0;
    GAMS = 0.0;
    HML = 0.0;
    HS = 0.0;
    T0 = 280.0;
    BVF = 0.01;
    depth = 8000.0;
    width = 8000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    aC = 5000.0;
    lC = 4000.0;
    mtnh = '250m';
    hC = 250.0;
    u0 = 10.0;
    uj = 0.0;
    b = 0.0;
end

%% Set up physical parameters for basic state(taken from Tempest defaults)
BS = struct('gam',gam,'Rd',Rd,'cp',cp,'cv',cv,'GAMT',GAMT,'HT',HT,'GAMS', ...
            GAMS,'HML',HML,'HS',HS,'ga',ga,'p0',p0,'T0',T0,'BVF',BVF);

%% Set up the jet and mountain profile parameters
UJ = struct('u0',u0,'uj',uj,'b',b,'ga',ga);
%UJ = struct('u0',u0,'uj1',uj1,'uj2',uj2,'ga',ga);
DS = struct('z0',z0,'zH',zH,'l1',l1,'l2',l2,'L',L,'aC',aC,'lC',lC,'hC',hC);

%% Set up the Rayleigh Layer with a coefficient one order of magnitude less than the order of the wave field
RAY = struct('depth',depth,'width',width,'nu1',nu1,'nu2',nu2,'nu3',nu3,'nu4',nu4);

%% Compute the LHS coefficient matrix and force vector for the test case
[LD,FF,REFS] = ...
computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NX, NZ, applyLateralRL);

%% Get the boundary conditions
[FFBC,SOL,sysDex] = GetAdjust4CBC(BC,NX,NZ,OPS,FF);

%% Solve the system using the matlab linear solver
%
disp('Solve the raw system with matlab default \.');
tic
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = FFBC(sysDex,1);
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
uxz = reshape(SOL((1:OPS)),NZ,NX);
whxz = reshape(SOL((1:OPS) + OPS),NZ,NX);
rxz = reshape(SOL((1:OPS) + 2*OPS),NZ,NX);
pxz = reshape(SOL((1:OPS) + 3*OPS),NZ,NX);

%% Convert \hat{w} to w using the reference density profile
wf = sqrt(REFS.rref0) * REFS.rref.^(-0.5);
wxz = wf .* whxz;
%wxz = whxz;

%% Interpolate to a regular grid using Hermite and Legendre transforms'
NXI = 500;
NZI = 300;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(pxz), NXI, NZI, 0, 0);

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

% INTERPOLATED GRID PLOTS
%% Compute the reference state initialization
if strcmp(TestCase,'ShearJetSchar') == true
    [lpref, lrref, dlpref, ~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT);
    [lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZLINT);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
elseif strcmp(TestCase,'ClassicalSchar') == true
    [lpref, lerref, ~, ~] = computeBackgroundPressureCBVF(BS, ZINT, speye(NZI));
    [ujref, ~] = computeJetProfileUniform(UJ, lpref);
elseif strcmp(TestCase,'NonhydroMtn') == true
    [lpref, lrref, ~, ~] = computeBackgroundPressureCBVF(BS, ZINT, speye(NZI));
    [ujref, ~] = computeJetProfileUniform(UJ, lpref);
end

fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); contourf(XI,ZI,ujref + uxzint,31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U (m/s)');
subplot(1,2,2); contourf(XI,ZI,wxzint,31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Vertical Velocity W (m/s)');

fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); contourf(XI,ZI,rxzint,31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Density (kg/m^3)');
subplot(1,2,2); contourf(XI,ZI,pxzint,31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Pressure (Pa)');

fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
contourf(XI,ZI,abs(uxzint) ./ abs(ujref),31,'LineStyle','none'); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Nonlinearity Ratio $\left| \frac{u}{\bar{u}} \right|$','FontSize',32,'FontWeight','normal','Interpreter','latex');
drawnow

%% Compute the local Ri number and plot ...
%
lrho = REFS.lrref + real(rxz);
dlrho = REFS.sig .* (REFS.DDZ * lrho);
uj = REFS.ujref + real(uxz);
duj = REFS.sig .* (REFS.DDZ * uj);
Ri = -ga * dlrho ./ (duj.^2);

RiREF = -BS.ga * REFS.dlrref(:,1);
RiREF = RiREF ./ (REFS.dujref(:,1).^2);

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
%{
subplot(1,2,1);
semilogx(Ri(:,xdex),1.0E-3*REFS.ZTL(:, xdex),'ks','LineWidth',1.5); grid on;
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',1.5);
hold off;
xlabel('Ri','FontSize',30);
ylabel('Altitude (km)','FontSize',30);
ylim([0.0 5.0]);
xlim([0.01 1.0E3]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
%drawnow;

subplot(1,2,2);
%}
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

pt = mtit(['Local Ri Number - ' mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

dirname = '../ShearJetSchar/';
fname = [dirname 'RI_TEST_' num2str(hC)];
drawnow;
screen2png(fname);
%}

%% Compute the local potential temperature and plot...
rho = exp(lrho);
lp = REFS.lpref + real(pxz);
p = exp(lp);
pt = p ./ (Rd * rho) .* (p0 * p.^(-1)).^(Rd / cp);
dpt = REFS.sig .* (REFS.DDZ * pt);
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

pt = mtit(['Convective Stability - ' mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

dirname = '../ShearJetSchar/';
fname = [dirname 'CONV_TEST_' num2str(hC)];
drawnow;
screen2png(fname);

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
%
close all;
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceHER' char(TestCase) int2str(hC) '_8KRL.mat'];
save(fileStore);
%}
