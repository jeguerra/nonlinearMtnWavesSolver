function [FFBC,SOL,sysDex] = GetAdjust4CBC_SSPRK53(BC,NX,NZ,OPS,FF)
    numVar = 4;
    SOL = 0.0 * FF;
    % Adjust for the boundary forcing
    FFBC = FF;

    %% Create boundary condition indices
    %
    ubdex = 1:NZ:OPS;
    wbdex = ubdex + OPS;
    rbdex = ubdex + 2*OPS;
    pbdex = ubdex + 3*OPS;
    BotOut = [ubdex wbdex rbdex pbdex];
    %
    utdex = NZ:NZ:OPS;
    wtdex = utdex + OPS;
    rtdex = utdex + 2*OPS;
    ptdex = utdex + 3*OPS;
    TopOut = [utdex wtdex rtdex ptdex];
    %
    uldex = 1:NZ;
    urdex = (NX-1)*NZ+1:OPS;
    wldex = uldex + OPS;
    wrdex = urdex + OPS;
    rldex = uldex + 2*OPS;
    rrdex = urdex + 2*OPS;
    pldex = uldex + 3*OPS;
    prdex = urdex + 3*OPS;
    %}
    %{
    uldex = 2:NZ-1;
    urdex = (NX-1)*NZ+2:OPS-1;
    wldex = uldex + OPS;
    wrdex = urdex + OPS;
    rldex = uldex + 2*OPS;
    rrdex = urdex + 2*OPS;
    pldex = uldex + 3*OPS;
    prdex = urdex + 3*OPS;
    %}
    LeftRightOut = [uldex urdex wldex wrdex rldex rrdex pldex prdex];
    RightOut = [urdex wrdex rrdex prdex];

    if BC == 0
        disp('Hermite-Laguerre BC');
        rowsOut = [];
    elseif BC == 1
        % Only apply the BC on W at the top
        disp('No-flux W TOP, Periodic Lateral');
        rowsOut = [utdex wtdex RightOut];% ubdex wbdex];
    elseif BC == 2
        % Apply BC on W and U for no wave flow across boundaries
        disp('No-flux W TOP, Dirichlet All LATERAL');
        rowsOut = [TopOut BotOut LeftRightOut];
    elseif BC == 3
        % Apply BC on W and U for no wave flow across boundaries and
        % thermodynamic variable vanish at lateral boundaries
        disp('BC on W and U for no outflow, rho and P vanish at lateral boundaries.');
        rowsOut = [wtdex uldex urdex wldex wrdex rldex rrdex pldex prdex];
    elseif BC == 4
        % Apply BC on W and U for no wave flow across boundaries and
        % thermodynamic variable vanish at lateral boundaries
        disp('BC on W and U for no outflow, rho and P vanish at all boundaries.');
        rowsOut = [wtdex uldex urdex rldex rrdex pldex prdex rtdex ptdex];
    else
        disp('ERROR: Invalid BC combination... applying default on W bottom only.');
        rowsOut = wtdex;
    end

    sysDex = setdiff(1:numVar*OPS, rowsOut);
end