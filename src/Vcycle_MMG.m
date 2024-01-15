function v = Vcycle_MMG(amg, f, v0, mu, lb, ub, pre_smoother, post_smoother)  
% solve equations Ae = r in each level
nlev = length(amg);
n = size(f,2);
amg(1).f = f;
amg(1).v = v0;

eigUpper = 1.01*computeMaxEigenvalue(amg(1).A, rand(size(amg(1).f)), 100); %eigs(A,1,'largestreal');

%% pre-smoothing
for i=1:nlev-1
  % mu steps pre-relax (e.g., forward G-S)
  % on the finest level use a constrained smoother
  if i==1
    amg(i).v = chebyshev_smoother(amg(i).A, amg(i).f,amg(i).v,lb, ub,mu,eigUpper);
    [~, activeSet] = ComputeProjGrad(amg(1).v, amg(1).A*amg(1).v - f, lb, ub);
    if ~isempty(activeSet)
      % reconstruct transfer operator if activeset is detected
      P_truncated = amg(1).P;
      P_truncated(activeSet,:) = 0 ;

      amg(2).A = P_truncated' * amg(1).A * P_truncated;
      % restrict residual to coarser level with truncated operator
      amg(2).f = P_truncated' * (amg(1).f - amg(1).A * amg(1).v);
      
      % recompute Galerkin projection
      for ilev = 2:numel(amg)-1
        amg(ilev+1).A = amg(ilev).P' * amg(ilev).A * amg(ilev).P;
      end
    else
      % restrict residual to coarser level
      amg(2).f = amg(1).P' * (amg(1).f - amg(1).A * amg(1).v);
    end
  else
    % on the other levels the smoother does not change
    amg(i).v = SSOR(amg(i).A, amg(i).f,amg(i).v,1, mu, 0, false );
    % restrict residual to coarser level
    amg(i+1).f = amg(i).P' * (amg(i).f - amg(i).A * amg(i).v);
  end
end

%% the coarsest level: direct solver
if (isfield(amg(nlev), 'fact') && ~isempty(amg(nlev).fact))
  R = amg(nlev).fact.R;
  s = amg(nlev).fact.s;
  f = amg(nlev).f;
  x(s,:) = R \(R'\f(s,:));
  amg(nlev).v = x;
else
  amg(nlev).v = amg(nlev).A \ amg(nlev).f;
end

% i = nlev;
% Mi = pre_smoother(amg(i).A);   amg(i).v = Mi \ amg(i).f;
% Mi = post_smoother(amg(i).A);  amg(i).v = amg(i).v + Mi \ (amg(i).f - amg(i).A * amg(i).v);

%% post-smoothing
for i=nlev-1:-1:1
  % prolongate to finer level
  amg(i).v = amg(i).v + amg(i).P * amg(i+1).v;
  % mu steps post_relax (e.g., backward G-S) with the above initial guess
  if i==1
    amg(i).v = chebyshev_smoother(amg(i).A, amg(i).f,amg(i).v,lb, ub,mu,eigUpper);
  else
    amg(i).v = SSOR(amg(i).A, amg(i).f,amg(i).v,1, mu, 0, false );
  end
end
v = amg(1).v;
end