function [Dii,xi_el,xi_gd] = computeDerivInt2Int(nel, nint)
    %% INPUTS
    % nel = number of elements in the column
    % nint = number of grid interfaces in the column
    %% OUTPUTS
    % Dii = column interfaces to interfaces derivative operator
    % xi_el = overlapping element grid (with redundant interfaces)
    % xi_gd = non-overlapping column grid that matches Dii
    
    %% Create the element wise and unique column grid on [0. 1.]
end