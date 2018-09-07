function [SOL,sysDex] = GetAdjust4CBC(REFS,BC,NX,NZ,OPS)
    %% Number of variables and block positions in the solution vector
    numVar = 4;
    iW = 1;
    iP = 2;
    iT = 3;

    %% Set an initial solution vector
    SOL = zeros(numVar*OPS,1);

    %% Create boundary condition indices
    uldex = 2:NZ-1;
    wldex = uldex + OPS;
    rldex = wldex + OPS;
    pldex = rldex + OPS;
    LeftOutExcludeCorners = [uldex wldex rldex pldex];
    RightOutExcludeCorners = LeftOutExcludeCorners + (OPS - NZ);
    
    uldex = 1:NZ;
    wldex = uldex + OPS;
    rldex = wldex + OPS;
    pldex = rldex + OPS;
    LeftWTOut = [uldex wldex rldex pldex];
    RightWTOut = LeftOutExcludeCorners + (OPS - NZ);
    
    utdex = NZ:NZ:OPS;
    wtdex = utdex + iW*OPS;
    rtdex = utdex + iP*OPS;
    ptdex = utdex + iT*OPS;
    TopOut = [utdex wtdex rtdex ptdex];
    
    ubdex = 1:NZ:(OPS - NZ + 1);
    wbdex = ubdex + iW*OPS;
    rbdex = ubdex + iP*OPS;
    pbdex = ubdex + iT*OPS;
    BotOut = [ubdex wbdex rbdex ptdex];
    
    if BC == 0
        disp('Hermite-Lagrange LogP-LogTheta Model, Dirichlet Lateral BC...');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        SOL(pbdex) = -REFS.ZTL(1,:) .* REFS.dlthref(1,:);
        
        % U and LnP need to be constrained at the corners
        %{
        BotLeftCorner = [uldex(1) rldex(1)];
        BotRightCorner = BotLeftCorner + (OPS - NZ);
        
        TopLeftCorner = [uldex(end) rldex(end)];
        TopRightCorner = TopLeftCorner + (OPS - NZ);
        
        rowsOut = [BotLeftCorner BotRightCorner ...
                   TopLeftCorner TopRightCorner ...
                   wtdex ptdex wbdex pbdex];
        %}
        rowsOut = [utdex wtdex ptdex wbdex pbdex];
           
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    elseif BC == 1
        disp('Hermite-Lagrange Rho-RhoTheta Model, Dirichlet Lateral BC...');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.rref(1,:) .* REFS.ujref(1,:);
        
        rowsOut = [utdex wtdex ptdex wbdex pbdex];
               
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    elseif BC == 2
        disp('Applying BC FFT-Lagrange Model, Periodic Lateral BC');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        ZTLF = fft(REFS.ZTL, NX, 2);
        SOL(pbdex) = -ZTLF(1,:) .* REFS.dlthref(1,:);
        
        rowsOut = [utdex wtdex ptdex wbdex pbdex];
               
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    elseif BC == 3
        disp('Hermite-Lagrange LogP-LogTheta Model, Transient Solve');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        SOL(pbdex) = -REFS.ZTL(1,:) .* REFS.dlthref(1,:);
        
        rowsOut = [utdex wtdex ptdex wbdex pbdex];
               
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    end
end