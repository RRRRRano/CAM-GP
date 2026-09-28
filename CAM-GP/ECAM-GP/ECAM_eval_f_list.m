function f = ECAM_eval_f_list(parv, dat, y, p, q, m, tau, n, rcoord, ...
                              lambda1, lambda2)
%ECAM_EVAL_F_LIST  Penalised negative log-likelihood of ECAM-GP.
%
%   F = ECAM_EVAL_F_LIST(PARV, DAT, Y, P, Q, M, TAU, N, RCOORD, LAMBDA1, LAMBDA2)
%
%   Same objective as CAM_eval_f_list, but the length-scale parameters of the
%   adjustment terms number sum(m) - q rather than p*sum(m).

    % Penalised parameters: variances and length scales, not the mixing weight.
    nVariance  = 2 + q;
    nPenalised = nVariance + 2*p + (sum(m) - q);

    if min(parv(1:nPenalised)) < 0
        f = 1e10;
        return
    end

    % ---- covariance matrix ---------------------------------------------
    covm = zeros(n);
    for i = 1:size(rcoord,1)
        covm(rcoord(i,1), rcoord(i,2)) = ECAM_covx_m_add( ...
            [dat(rcoord(i,1),:), dat(rcoord(i,2),:)], parv, p, q, m);
    end
    covm = covm + covm';
    for i = 1:n
        covm(i,i) = ECAM_covx_add(dat(i,:), dat(i,:), parv, p, q, m) + tau;
    end

    % ---- Cholesky factorisation ----------------------------------------
    [Tm, flag] = chol(covm);
    if flag > 0
        covm = covm + 1e-4 * eye(n);
        [Tm, flag] = chol(covm);
        if flag > 0
            f = 1e10;
            return
        end
    end
    logdet = 2 * sum(log(diag(Tm)));

    invT = inv(Tm);
    invc = invT * invT';
    one  = ones(n, 1);

    % ---- profile negative log-likelihood -------------------------------
    term1 = logdet;
    term2 = y' * invc * y;
    term3 = (one' * invc * y)^2 / sum(invc(:));
    likelihood = term1 + term2 - term3;

    % ---- group-wise L2 penalties ---------------------------------------
    varianceParams = parv(1:nVariance);
    scaleParams    = parv(nVariance+1 : nPenalised);
    reg = lambda1 * sum(varianceParams.^2) + lambda2 * sum(scaleParams.^2);

    f = likelihood + reg;
end
