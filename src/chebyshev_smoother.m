function [x, activeSet, ProjResNorm] = chebyshev_smoother(A, b, x, lb, ub, maxIter, eigUpper)

eigLower = 0.06*eigUpper;

% eigMax = eigs(A,1,'LM','Tolerance',1e-10);
% eigMin = eigs(A,1,'SM','Tolerance',1e-10);

theta = (eigUpper + eigLower) / 2.0;
delta = (eigUpper - eigLower) / 2.0;

res = b - A*x;

iter = 0;
while iter < maxIter
  if iter == 0
    p = res;
    alpha = 1.0 / theta;
  elseif iter == 1
    beta = 0.5 * (delta * alpha)^2;
    alpha = 1.0 / (theta - beta / alpha);
    p = res + beta * p;
  else
    beta = (0.5 * delta * alpha)^2;
    alpha = 1.0 / (theta - beta / alpha);
    p = res + beta * p;
  end
  x = x + alpha * p;
  
  % Project the iterate to feasible set
  x = min(max(x, lb), ub);

  res = b - A * x;

  [ProjGrad, activeSet] = ComputeProjGrad(x, -res, lb, ub);
  %nActiveSet = numel(activeSet)

  ProjResNorm = norm(ProjGrad);
  iter = iter + 1;
end
end