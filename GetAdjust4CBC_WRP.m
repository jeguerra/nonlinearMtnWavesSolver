function [FFBC,SOL,sysDex] = GetAdjust4CBC_WRP(BC,NX,NZ,OPS,FF)
    numVar = 3;
    SOL = 0.0 * FF;
    % Adjust for the boundary forcing
    FFBC = FF;

    %% Create boundary condition indices
    %
    wbdex = 1:NZ:OPS;
    rbdex = wbdex + OPS;
    pbdex = wbdex + 2*OPS;
    BotOut = [wbdex rbdex pbdex];
    %
    wtdex = NZ:NZ:OPS;
    rtdex = wtdex + OPS;
    ptdex = wtdex + 2*OPS;
    TopOut = [wtdex rtdex ptdex];
    %
    wldex = 1:NZ;
    wrdex = (NX-1)*NZ+1:OPS;
    rldex = wldex + OPS;
    rrdex = wrdex + OPS;
    pldex = wldex + 2*OPS;
    prdex = wrdex + 2*OPS;
    %}
    LeftRightOut = [wldex wrdex rldex rrdex pldex prdex];
    RightOut = [wrdex rrdex prdex];

    if BC == 0
        disp('No Specific BC`s Set');
        rowsOut = [];
    elseif BC == 1
        % Only apply the BC on W at the top
        disp('No-flux W TOP, Periodic Lateral');
        %rowsOut = wtdex;
        rowsOut = [wtdex RightOut];
    elseif BC == 2
        % Apply BC on W and U for no wave flow across boundaries
        disp('No-flux W TOP, Dirichlet All LATERAL');
        rowsOut = [wtdex LeftRightOut];
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