function [FFBC,SOL,sysDex] = GetAdjust4CBC(BC,NX,NZ,OPS,FF)
    numVar = 4;
    SOL = 0.0 * FF;
    % Adjust for the boundary forcing
    FFBC = FF;

    %% Create boundary condition indices
    %
    uldex = 2:NZ-1;
    wldex = uldex + OPS;
    rldex = uldex + 2*OPS;
    pldex = uldex + 3*OPS;
    %}
    LeftOutExcludeCorners = [uldex wldex rldex pldex];
    RightOutExcludeCorners = LeftOutExcludeCorners + (OPS - NZ);
    
    if BC == 0
        disp('Hermite-Lagrange Model, Dirichlet Lateral BC...');
        var1BotLeftCorner = 1;
        var1TopLeftCorner = NZ;
        var1BotRightCorner = OPS - NZ + 1;
        var1TopRightCorner = OPS;
        var3BotLeftCorner = []; %(2*OPS + 1);
        var3TopLeftCorner = []; %(3*OPS - NZ + 1);
        var3BotRightCorner = []; %(2*OPS + NZ);
        var3TopRightCorner = []; %(3*OPS);
        rowsOut = [LeftOutExcludeCorners ...
                   RightOutExcludeCorners ...
                   var1BotLeftCorner ...
                   var1TopLeftCorner ...
                   var1BotRightCorner ...
                   var1TopRightCorner ...
                   var3BotLeftCorner ...
                   var3TopLeftCorner ...
                   var3BotRightCorner ...
                   var3TopRightCorner];
    elseif BC == 1
        disp('Applying BC FFT-Lagrange Model...');
        rowsOut = [];
    elseif BC == 2
        disp('Hermite-Lagrange Model, Free Lateral BC...');
        var1BotLeftCorner = 1;
        var1TopLeftCorner = NZ;
        var1BotRightCorner = OPS - NZ + 1;
        var1TopRightCorner = OPS;
        var3BotLeftCorner = (2*OPS + 1);
        var3TopLeftCorner = (3*OPS - NZ + 1);
        var3BotRightCorner = (2*OPS + NZ);
        var3TopRightCorner = (3*OPS);
        rowsOut = [var1BotLeftCorner ...
                   var1TopLeftCorner ...
                   var1BotRightCorner ...
                   var1TopRightCorner ...
                   var3BotLeftCorner ...
                   var3TopLeftCorner ...
                   var3BotRightCorner ...
                   var3TopRightCorner];
    elseif BC == 3
        disp('Hermite-Lagrange Model, Dirichlet Left BC...');
        var1BotLeftCorner = 1;
        var1TopLeftCorner = NZ;
        var1BotRightCorner = OPS - NZ + 1;
        var1TopRightCorner = OPS;
        var3BotLeftCorner = (2*OPS + 1);
        var3TopLeftCorner = (3*OPS - NZ + 1);
        var3BotRightCorner = (2*OPS + NZ);
        var3TopRightCorner = (3*OPS);
        rowsOut = [LeftOutExcludeCorners ...
                   var1BotLeftCorner ...
                   var1TopLeftCorner ...
                   var1BotRightCorner ...
                   var1TopRightCorner ...
                   var3BotLeftCorner ...
                   var3TopLeftCorner ...
                   var3BotRightCorner ...
                   var3TopRightCorner];
    end

    sysDex = setdiff(1:numVar*OPS + 2*NX, rowsOut);
end