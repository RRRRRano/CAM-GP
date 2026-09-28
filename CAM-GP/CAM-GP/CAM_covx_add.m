function res = CAM_covx_add(w1, w2, parv, p, q, m)
%CAM_COVX_ADD  Covariance between two inputs under CAM-GP.
%
%   RES = CAM_COVX_ADD(W1, W2, PARV, P, Q, M)
%
%   W1 and W2 are 1-by-(p+q) inputs, ordered as continuous variables followed by
%   categorical levels coded 1..m(h).
%
%   PARV holds the model parameters:
%     parv(1)                     sigma_0a^2  baseline additive variance
%     parv(2)                     sigma_0m^2  baseline multiplicative variance
%     parv(3 : 2+q)               sigma_h^2   adjustment-term variances
%     parv(3+q : 2+q+p)           theta_0a    baseline additive length scales
%     parv(3+q+p : 2+q+2p)        theta_0m    baseline multiplicative length scales
%     parv(2+q+2p+1 : end-1)      adjustment length scales, p per level
%     parv(end)                   alpha_raw   unconstrained mixing weight
%
%   The baseline kernel is the weighted sum of a dimension-decoupled additive
%   kernel and a multiplicative kernel (Eq. 21 of the paper).  An adjustment
%   term contributes only when the two inputs share the level of the
%   corresponding categorical variable, and it mixes the additive and
%   multiplicative aggregations with the weight alpha (Eq. 23).

    % transpose so that x1 and x2 are p-by-1 column vectors: this keeps
    % theta .* diffSq an elementwise operation on p-by-1 vectors
    x1 = w1(1:p).';
    x2 = w2(1:p).';
    z1 = w1(p+1:end);
    z2 = w2(p+1:end);

    diffSq = (x1 - x2).^2;

    % ---- baseline kernel, Eq. 21 ----------------------------------------
    sigma0a = parv(1);
    sigma0m = parv(2);
    theta0a = parv(3+q   : 2+q+p);
    theta0m = parv(3+q+p : 2+q+2*p);

    baseAdd  = mean(exp(-theta0a .* diffSq));      % 1/p normalisation
    baseMult = exp(-sum(theta0m .* diffSq));

    res = sigma0a * baseAdd + sigma0m * baseMult;

    % ---- mixing weight, alpha in (0,1) ----------------------------------
    alpha = 1 / (1 + exp(-parv(end)));

    % ---- adjustment terms, Eq. 23 ---------------------------------------
    idxAdj = 2 + q + 2*p + 1;          % first adjustment length scale
    cursor = idxAdj;

    for h = 1:q
        sigmaH = parv(2 + h);

        if z1(h) ~= z2(h)
            cursor = cursor + p * m(h);    % levels differ: no contribution
            continue
        end

        l        = z1(h);
        startIdx = cursor + (l - 1) * p;
        thetaHL  = parv(startIdx : startIdx + p - 1);

        addPart  = mean(exp(-thetaHL .* diffSq));
        multPart = exp(-sum(thetaHL .* diffSq));

        res = res + sigmaH * (alpha * addPart + (1 - alpha) * multPart);

        cursor = cursor + p * m(h);
    end
end
