% Plots the Classical Schar case from classical theory and my reference
% solution

clc
clear
GRS = load('100X80SpectralReferenceHERClassicalSchar250.mat');
CTS = load('AnalyticalSchar.mat');

cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);

fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
cmr = colormap(cmap);
HT = 25.0;

%% Plot the Guellrich reference solution
s1 = subplot(1,2,1); contourf(1.0E-3 * GRS.XI, 1.0E-3 * GRS.ZLINT, GRS.wxzint, 41); %colorbar; 
grid on;
caxis([-2.0 2.0]);
xlim([-20.0 20.0]);
ylim([0.0 HT]);
title('Reference Solution W (m/s)','FontWeight','normal','Interpreter','tex');
ylabel('Altitude (km)');
xlabel('Horizontal Distance (km)');
fig.CurrentAxes.FontSize = 24;
fig.CurrentAxes.LineWidth = 1.5;

%% Plot the classical Schar solution
s2 = subplot(1,2,2); contourf(1.0E-3 * CTS.X, 1.0E-3 * CTS.Z, CTS.w',41); colorbar; grid on;
caxis([-2.0 2.0]);
xlim([-20.0 20.0]);
ylim([0.0 HT]);
title('Classical Theory W (m/s)','FontWeight','normal','Interpreter','tex');
xlabel('Horizontal Distance (km)');
fig.CurrentAxes.FontSize = 24;
fig.CurrentAxes.LineWidth = 1.5;

s1Pos = s1.Position;
s2.Position(3:4) = s1Pos(3:4);
dirname = '../ShearJetSchar/';
fname = [dirname 'ClassicalScharCompare'];
drawnow;
screen2png(fname);

%% Plot the boundary
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
WLT = CTS.w';
plot(GRS.XI(1,:), GRS.wxzint(1,:),'s-',CTS.X(1,:), WLT(1,:),'o-');