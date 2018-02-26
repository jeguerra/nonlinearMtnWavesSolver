function [FFBC,SOL,sysDex] = GetAdjust4CBC(BC,NX,NZ,OPS,FF)
    numVar = 4;
    SOL = 0.0 * FF;
    % Adjust for the boundary forcing
    FFBC = FF;

    %% Create boundary condition indices
    %
    ubdex = 1:NZ:(OPS - NZ + 1);
    wbdex = ubdex + OPS;
    rbdex = ubdex + 2*OPS;
    pbdex = ubdex + 3*OPS;
    BotOut = [ubdex wbdex rbdex pbdex];
    %}
    utdex = NZ:NZ:OPS;
    wtdex = utdex + OPS;
    rtdex = utdex + 2*OPS;
    ptdex = utdex + 3*OPS;
    TopOut = [utdex wtdex rtdex ptdex];
    %
    uldex = 1:NZ;
    urdex = (OPS - NZ + 1):OPS;
    wldex = uldex + OPS;
    wrdex = urdex + OPS;
    rldex = uldex + 2*OPS;
    rrdex = urdex + 2*OPS;
    pldex = uldex + 3*OPS;
    prdex = urdex + 3*OPS;
    %}
    LeftRightOut = [uldex urdex wldex wrdex rldex rrdex pldex prdex];
    LeftOut = [uldex wldex rldex pldex];
    RightOut = [urdex wrdex rrdex prdex];

    if BC == 0
        disp('All quantities vanish at the upstream lateral boundary');
        rowsOut = LeftOut;
    elseif BC == 1
        disp('All quantities vanish at the top and lateral boundary');
        rowsOut = [TopOut LeftRightOut];
    elseif BC == 2
        disp('Dirichlet All Top');
        rowsOut = TopOut;
    elseif BC == 3
        disp('Dirichlet W Top, Bottom, and Lateral');
        rowsOut = [ubdex utdex wbdex wtdex LeftRightOut];
    elseif BC == 4
        disp('Dirichlet Testing');
        rowsOut = [wtdex];
    else
        disp('Default BC... nothing removed');
        rowsOut = [];
    end

    sysDex = setdiff(1:numVar*OPS + 2*NX, rowsOut);
end