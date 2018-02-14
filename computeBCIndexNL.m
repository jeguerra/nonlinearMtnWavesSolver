function sysDex = computeBCIndexNL(BC,NX,NZ,OPS)
    numVar = 4;

    %% Create boundary condition indices
    %{
    ubdex = 1:NZ:OPS;
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
    RightOut = [urdex wrdex rrdex prdex];

    if BC == 0
        disp('Dirichlet W Top, All Lateral');
        rowsOut = [wtdex LeftRightOut];
    elseif BC == 1
        disp('Dirichlet W Top Only');
        rowsOut = wtdex;
    elseif BC == 2
        disp('Dirichlet All Top');
        rowsOut = TopOut;
    elseif BC == 3
        disp('Dirichlet All Top and Lateral');
        rowsOut = [TopOut LeftRightOut];
    elseif BC == 4
        disp('Dirichlet W Top, Periodic Lateral');
        rowsOut = [wtdex RightOut];
    else
        disp('ERROR: Invalid BC combination... applying default on W bottom only.');
        rowsOut = wtdex;
    end

    sysDex = setdiff(1:numVar*OPS, rowsOut);
end