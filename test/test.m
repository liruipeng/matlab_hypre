clear;
close all;

addpath ../src ../mex

% 2 - D / 3 - D (shifted) Laplacian
nx = 128;
ny = 128;
nz = 1;
shift = 0.0;
A = fd3d(nx, ny, nz, 0, 0, 0, shift);
n = size(A, 1);

% exact solution and rhs
u = rand(n, 1);
f = A * u;

%% --------- Setup Phase: build AMG hierarchy
% hypre parameters
setup.print_level  =  3;
setup.coarsen_type =  10;
setup.relax_type   =  3;
setup.relax_sweeps =  1;
setup.max_level    = 20;
setup.sol_tol      = 1e-8;
setup.max_iter     = 100;
setup.solver_id    = 0; % 0: AMG 1: AMG + CG
% amg is an array of struct that has fields {cf, A, P, v, f}
% amg(i) for level i of the hierarchy
% amg(i).{cf, A, P} are computed by hypre (sequential, i.e., NP = 1)
% cf: C_PT = 1; F_PT = -1
% amg(i).{v, f} are work space for solve phase
[amg, y] = hypre_amg(A, f, setup);

%% -------- Choose smoothers
smoother = 'G';
if (smoother(1) == 'G')
   % G - S
   pre_smoother  = @(A, b) tril(A) \ b;
   post_smoother = @(A, b) triu(A) \ b;
else
   % weighted Jacobi
   w = 2 / 3;
   pre_smoother  = @(A, b) 1 / w * diag(diag(A)) \ b;
   post_smoother = pre_smoother
end
% num of relax steps
mu = 1;

%% -------- Solve Phase: V - cycle
fprintf(1, '- - - - - - -  AMG SOLVE PHASE (MATLAB) - - - - - - - - - -\n');

tol    = setup.sol_tol;
maxits = setup.max_iter;

% initial guess
v = zeros(n, 1);
r = f - A * v;
normr = norm(r);
err(1) = norm(u - v);
res(1) = normr;
% tolr = tol * normr;
iter = 0;
fprintf(1, 'Cycle    residual         relative residual\n')
fprintf(1, '%5d    %.6e     %.6e\n', 0, res(1), res(1) / res(1));
while (normr / res(1) > tol && iter < maxits)
   iter = iter + 1;
   v = Vcycle(amg, f, v, mu, pre_smoother, post_smoother);
   r = f - A * v;
   normr = norm(r);
   res(iter + 1) = normr;
   err(iter + 1) = norm(u - v);
   fprintf(1, '%5d    %.6e     %.6e\n', iter, res(iter + 1), res(iter + 1) / res(1));
end
fprintf(1, 'Iter %d, Relres %e [res %e], Err %e \n', iter, res(end) / res(1), res(end), err(end));
%%
%figure;
%subplot(2, 1, 1); semilogy(res, '--ro'); title('Residue norms');
%subplot(2, 1, 2); semilogy(err, '--bx'); title('Error norms');

