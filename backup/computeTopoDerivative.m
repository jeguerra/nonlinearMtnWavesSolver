function [terrain,terrDeriv] = computeTopoDerivative(TestCase,xh,DS)
    ht = 0.0 * xh;
    dhdx = 0.0 * xh;

    %% Get the correct mountain for this test
    if ((strcmp(TestCase,'ShearJetSchar') == true) || ...
            (strcmp(TestCase,'ShearJetScharCBVF') == true) || ...
            (strcmp(TestCase,'ClassicalSchar') == true))
        ht = DS.hC * exp(-xh.^2/DS.aC^2) .* (cos(pi * xh / DS.lC)).^2;
       
        dhdx = -DS.hC * exp(-xh.^2/DS.aC^2) .* ( ...
            2.0 * xh / (DS.aC^2) .* (cos(pi * xh / DS.lC)).^2 + ...
            pi/DS.lC * sin(2.0 * pi * xh / DS.lC));
    elseif ((strcmp(TestCase,'HydroMtn') == true) || (strcmp(TestCase,'NonhydroMtn') == true))
        ht = DS.hC * (1.0 + xh.^2 / DS.aC).^(-1);
        
        dhdx = -DS.hC * (2 * xh / DS.aC) .* (1.0 + xh.^2 / DS.aC).^(-2);
    elseif (strcmp(TestCase,'AndesMtn') == true)
        AM = load('/Users/TempestGuerra/Desktop/ShearFlowMountainWavesDATA/EcuadorAndesProfiles.mat');
        xinp = AM.xtest;
        tpinp = AM.mtest;
        
        ht = zeros(size(tpinp,1),length(xh));
        dhdx = ht;
        scale = 2 * pi / (max(xh) - min(xh));
        xhint = scale * (xh - min(xh));
        for ii=1:size(tpinp,1)
            % Interpolate the terrain height field
            ht(ii,:) = DS.hC * fourint(tpinp(ii,:), xhint);
            % Fourier terrain slopes at the native resolution
            dhdx30m = DS.hC * scale * fourdifft(tpinp(ii,:),1);
            % Sample the terrain derivatives at the interpolated grid
            dhdx(ii,:) = fourint(dhdx30m, xhint);
        end
        
        %plot(xinp, tpinp); figure;
        %plot(xh, ht); figure;
        %plot(xh, dhdx); pause;
        
        % Pick one of the profiles... CHANGE THIS
        ht = ht(3,:);
        dhdx = dhdx(3,:);
    end
    
    terrain = ht;
    terrDeriv = dhdx;
end