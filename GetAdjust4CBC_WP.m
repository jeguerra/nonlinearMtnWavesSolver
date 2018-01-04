function [FFBC,SOL,sysDex] = GetAdjust4CBC_WP(BC,NX,NZ,OPS,FF)
    numVar = 2;
    SOL = 0.0 * FF;
    % Adjust for the boundary forcing
    FFBC = FF;

    %% Create boundary condition indices
    %
    wbdex = 1:NZ:OPS;
    pbdex = wbdex + OPS;
    BotOut = [wbdex pbdex];
    %
    wtdex = NZ:NZ:OPS;
    ptdex = wtdex + OPS;
    TopOut = [wtdex ptdex];
    %
    wldex = 1:NZ;
    wrdex = (NX-1)*NZ+1:OPS;
    pldex = wldex + OPS;
    prdex = wrdex + OPS;
    %}
    LeftRightOut = [wldex wrdex pldex prdex];
    RightOut = [wrdex prdex];

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
        rowsOut = [wtdex wldex wrdex pldex prdex];
    elseif BC == 4
        % Apply BC on W and U for no wave flow across boundaries and
        % thermodynamic variable vanish at lateral boundaries
        disp('BC on W and U for no outflow, rho and P vanish at all boundaries.');
        rowsOut = [wtdex rldex pldex prdex ptdex];
    else
        disp('ERROR: Invalid BC combination... applying default on W bottom only.');
        rowsOut = wtdex;
    end

    sysDex = setdiff(1:numVar*OPS, rowsOut);
end