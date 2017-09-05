function [ujref,dujref] = computeJetProfileCBC(xi, prs)
% Computes the background sheared jet

    %% Compute the decay portion of the jet profile
    lu = log(xi + 1.0);
    ujref = prs.uj1 * lu .* exp(-prs.uj2 * lu) + prs.u0;
    
    %% Compute the shear in xi
    dujref = prs.uj1 * ((xi + 1.0).^(-1) .* exp(- prs.uj2 * lu) + ...
                       lu .* exp(- prs.uj2 * lu) .* (-prs.uj2 ./ (xi + 1.0)));
end

