function rmse = CAM_rmse(parv, xn, yn, dat, y, rcoord, p, q, tau, m, n)
%CAM_RMSE  Test-set root-mean-square error of a fitted CAM-GP model.
%
%   RMSE = CAM_RMSE(PARV, XN, YN, DAT, Y, RCOORD, P, Q, TAU, M, N)
%
%   Predicts the n2 test inputs XN with the model defined by PARV and fitted on
%   (DAT, Y), and returns the root-mean-square error against YN.  The constant
%   mean is estimated by generalised least squares, as in Algorithm 1 of the
%   paper.

    % ---- covariance matrix of the training data ------------------------
    covm = zeros(n);
    for i = 1:size(rcoord,1)
        covm(rcoord(i,1), rcoord(i,2)) = CAM_covx_m_add( ...
            [dat(rcoord(i,1),:), dat(rcoord(i,2),:)], parv, p, q, m);
    end
    covm = covm + covm';
    for i = 1:n
        covm(i,i) = CAM_covx_add(dat(i,:), dat(i,:), parv, p, q, m) + tau;
    end

    invc = inv(covm);
    one  = ones(n, 1);
    mu   = (one' * invc * y) / sum(invc(:));

    % ---- prediction at the test inputs ---------------------------------
    dif = zeros(size(yn,1), 1);
    for k = 1:size(yn,1)
        covv = zeros(n, 1);
        for j = 1:n
            covv(j) = CAM_covx_add(xn(k,:), dat(j,:), parv, p, q, m);
        end
        pred   = mu + covv' * invc * (y - mu * one);
        dif(k) = pred - yn(k);
    end

    rmse = sqrt(mean(dif.^2));
end
