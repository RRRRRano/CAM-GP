function res = ECAM_covx_m_add(w12, parv, p, q, m)
%ECAM_COVX_M_ADD  Covariance for a stacked pair of inputs.
%
%   RES = ECAM_COVX_M_ADD(W12, PARV, P, Q, M)
%
%   W12 = [W1, W2] is a 1-by-2(p+q) row vector holding two inputs, so that the
%   covariance matrix can be filled with a single call per index pair.

    w1 = w12(1 : p+q);
    w2 = w12(p+q+1 : 2*p+2*q);

    res = ECAM_covx_add(w1, w2, parv, p, q, m);
end
