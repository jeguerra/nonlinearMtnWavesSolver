function [terrain,terrDeriv] = computeTopoDerivative(TestCase,xh,DS)
    ht = 0.0 * xh;
    dhdx = 0.0 * xh;

    %% Get the correct mountain for this test
    if ((strcmp(TestCase,'ShearJetSchar') == true) || (strcmp(TestCase,'ClassicalSchar') == true))
        ht = DS.hC * exp(-xh.^2/DS.aC^2) .* (cos(pi * xh / DS.lC)).^2;
        
        dhdx = -2.0 * DS.hC * exp(-xh.^2/DS.aC^2) .* cos(pi * xh / DS.lC) .*...
           (pi/DS.lC * sin(pi * xh / DS.lC) + xh / (DS.aC^2) .* cos(pi * xh / DS.lC));
    elseif ((strcmp(TestCase,'HydroMtn') == true) || (strcmp(TestCase,'NonhydroMtn') == true))
        ht = DS.hC * (1.0 + xh.^2 / DS.aC).^(-1);
        
        dhdx = -DS.hC * (2 * xh / DS.aC) .* (1.0 + xh.^2 / DS.aC).^(-2);
    end
    
    terrain = ht;
    terrDeriv = dhdx;
end