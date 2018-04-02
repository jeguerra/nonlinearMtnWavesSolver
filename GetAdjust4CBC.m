function [SOL,sysDex] = GetAdjust4CBC(BS,REFS,BC,NX,NZ,OPS)
    %% Number of variables and block positions in the solution vector
    numVar = 4;
    iP = 1;
    iW = 2;
    iT = 3;

    %% Set an initial solution vector
    SOL = zeros(numVar*OPS,1);

    %% Create boundary condition indices
    uldex = 2:NZ-1;
    rldex = uldex + OPS;
    wldex = rldex + OPS;
    pldex = wldex + OPS;
    LeftOutExcludeCorners = [uldex wldex rldex pldex];
    RightOutExcludeCorners = LeftOutExcludeCorners + (OPS - NZ);
    
    utdex = NZ:NZ:OPS;
    rtdex = utdex + OPS;
    wtdex = rtdex + OPS;
    ptdex = wtdex + OPS;
    %TopOut = [utdex wtdex rtdex ptdex];
    
    ubdex = 1:NZ:(OPS - NZ + 1);
    wbdex = ubdex + iW * OPS;
    
    LeftCorners = [1 (iP*OPS + 1) (iT*OPS + 1) ...
                       NZ (iP*OPS + NZ) (iT*OPS + NZ)];
    RightCorners = [(OPS - NZ + 1) (iT*OPS - NZ + 1) (numVar*OPS - NZ + 1) ...
                       OPS iT*OPS numVar*OPS];
    
    if BC == 0
        disp('Hermite-Lagrange LogP-LogTheta Model, Dirichlet Lateral BC...');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        rowsOut = [LeftOutExcludeCorners RightOutExcludeCorners ...
                   LeftCorners RightCorners ...
                   wtdex wbdex];
               
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    elseif BC == 1
        disp('Hermite-Lagrange Rho-RhoTheta Model, Dirichlet Lateral BC...');
        rowsOut = [LeftOutExcludeCorners LeftCorners ...
                   RightOutExcludeCorners RightCorners];
        sysDex = setdiff(1:numVar*OPS + NX, rowsOut);
    elseif BC == 2
        disp('Applying BC FFT-Lagrange Model, Periodic Lateral BC');
        rowsOut = wtdex;
        sysDex = setdiff(1:numVar*OPS + NX, rowsOut);
    elseif BC == 3
        disp('Hermite-Lagrange LogP-LogTheta Model, Transient Solve');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        rowsOut = [LeftOutExcludeCorners RightOutExcludeCorners ...
                   LeftCorners RightCorners ...
                   wtdex wbdex];
               
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    end
end