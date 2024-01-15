function x = SSOR(A,b,x,maxIter,varargin)

% a minor modification to SSOR:
% when the constrained are passed to coarselevels
% matrix A_c = RAP can become indefinite
% as P_trucated is not a have a full rank

res = b - A*x;

activeSet = find(res==0);
omega = 1;

if ~isempty(activeSet)
  D = diag(A);
  D(activeSet) = 1;
  D = diag(D);
else
  D = diag(diag(A));
end

L = tril(A, -1);
U = triu(A, 1);

iter = 0 ;

while iter < maxIter
  iter = iter + 1;

  x = (D + omega*L) \ (omega*b - (omega*U + (omega - 1) * D) * x);
  x = (D + omega*U) \ (omega*b - (omega*L + (omega - 1) * D) * x);

end

end