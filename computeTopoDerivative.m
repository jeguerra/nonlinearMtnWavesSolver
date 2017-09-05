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
        
        %plot(xh, dhdx); pause;
        %{
        xff = linspace(min(xh), max(xh), length(xh));
        ht_fft = DS.hC * exp(-xff.^2/DS.aC^2) .* (cos(pi * xff / DS.lC)).^2;
        
        L = (max(xh) - min(xh));
        scale = 2 * pi / L;
        dhdx_fft = scale * fourdifft(ht_fft,1);
        
        plot(xh, dhdx, xff, dhdx_fft); pause;
        %}
    elseif ((strcmp(TestCase,'HydroMtn') == true) || (strcmp(TestCase,'NonhydroMtn') == true))
        ht = DS.hC * (1.0 + xh.^2 / DS.aC).^(-1);
        
        dhdx = -DS.hC * (2 * xh / DS.aC) .* (1.0 + xh.^2 / DS.aC).^(-2);
    elseif (strcmp(TestCase,'AndesMtn') == true)
        AM = load('/Users/TempestGuerra/Desktop/ShearFlowMountainWavesDATA/EcuadorAndesProfiles.mat');
        xinp = AM.xtest;
        tpinp = AM.mtest;
        
        ht = zeros(size(tpinp,1),length(xh));
        dhdx = ht;
        
        % Compute length scales and ratios for different domains
        Linp = 0.5 * (max(xinp) - min(xinp));
        L = 0.5 * (max(xh) - min(xh));
        dl = (L / Linp);
        scale = pi / L;
        
        % Compute the scaled interpolation grid [0 2pi]
        xhint = scale * (xh - min(xh));
        xhint = dl * xhint + pi * (1 - dl);
        
        dx = mean(diff(xinp));
        hsize = length(xinp);
        for ii=1:size(tpinp,1)
            % Interpolate the terrain height field
            ht(ii,:) = DS.hC * fourint(tpinp(ii,:), xhint);
            % Fourier terrain slopes at the native resolution
            dhdx30m = DS.hC * pi / Linp * fourdifft(tpinp(ii,:),1);
            % First order slopes for comparison checks
            topo30m = DS.hC * tpinp(ii,:);
            dhdx30m_O1 = topo30m;
            for jj=1:hsize
               if (jj == 1)
                   dhdx30m_O1(jj) = (topo30m(jj+1) - topo30m(jj)) / dx;
               elseif (jj == hsize)
                   dhdx30m_O1(jj) = (topo30m(jj) - topo30m(jj-1)) / dx;
               else
                   dhdx30m_O1(jj) = (topo30m(jj+1) - topo30m(jj-1)) / (2*dx);
               end
            end
            % Sample the terrain derivatives at the interpolated grid
            dhdx(ii,:) = fourint(dhdx30m, xhint);
            
            % DEBUG PLOTS WITH MULTIPLE COMPARISONS
            %{
            if ii == 3
                plot(xh, ht(ii,:), 's-', xinp, DS.hC * tpinp(ii,:), 'o-');
                figure;
                plot(xinp, dhdx30m, 's-', xinp, dhdx30m_O1, 'o-', xh, dhdx(ii,:), '+-');
                grid on;
                xlim([-1.0E4 1.0E4]);
                pause;
            end
            %}
        end
                
        % Pick one of the profiles... CHANGE THIS TO BE AUTOMATED
        ht = ht(3,:);
        dhdx = dhdx(3,:);
    end
    
    terrain = ht;
    terrDeriv = dhdx;
end