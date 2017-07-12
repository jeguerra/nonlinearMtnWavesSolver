% Plots the Classical Schar case from classical theory and my reference
% solution

clc
clear
GRS = load('100X80SpectralReferenceHERClassicalSchar250_8KRL.mat');
CTS = load('AnalyticalSchar_8KRL.mat');

cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);

fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
cmr = colormap(cmap);
HT = 25.0;

%% Plot the Guellrich reference solution
subplot(1,2,1); contourf(1.0E-3 * GRS.XI, 1.0E-3 * GRS.ZLINT, GRS.wxzint, 41); %colorbar; 
grid on;
caxis([-2.0 2.0]);
xlim(1.0E-3 * [GRS.l1 GRS.l2]);
ylim([0.0 HT]);
title('Reference Solution W (m/s)');
fig.CurrentAxes.FontSize = 30;
fig.CurrentAxes.LineWidth = 1.5;

%% Plot the classical Schar solution
subplot(1,2,2); contourf(1.0E-3 * CTS.X, 1.0E-3 * CTS.Z, fftshift(CTS.w',2),41); colorbar; grid on;
caxis([-2.0 2.0]);
xlim(1.0E-3 * [GRS.l1 GRS.l2]);
ylim([0.0 HT]);
title('Classical Theory W (m/s)');
set(gca,'FontSize', 30);
fig.CurrentAxes.FontSize = 30;
fig.CurrentAxes.LineWidth = 1.5;

dirname = '../ShearJetSchar/';
fname = [dirname 'ClassicalScharCompare'];
drawnow;
screen2png(fname);

%% Plot the boundary
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
WLT = fftshift(CTS.w',2);
plot(GRS.XI(1,:), GRS.wxzint(1,:),'s-',CTS.X(1,:), WLT(1,:),'o-');