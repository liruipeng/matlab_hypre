clear;
close all;

addpath ../src ../mex ../data


for myTest = 1:3
  str = sprintf('%d', myTest);
  loadfilename=strcat('Example_',str,'.mat');

  load(loadfilename)
  if myTest == 1
    % K = 1;
    fprintf('--------------------------------------------------------------\n')
    fprintf('Diffusion problem with obstacle [120 x 120]\n')
  elseif myTest == 2
    % K = [1e-6 0;
    %      0    1];
    fprintf('--------------------------------------------------------------\n')
    fprintf('Higly anisotropic Diffusion problem with obstacle [400 x 400]\n')
 elseif myTest == 3
    % K = [1e-4 0;
    %      0    1];
    fprintf('--------------------------------------------------------------\n')
    fprintf('Higly anisotropic Diffusion problem with obstacle unstructured mesh\n')
  end


  %A = fd3d(nx, ny, nz, 0, 0, 0, shift);
  n = size(A, 1);
  z = zeros(n,1);

  % exact solution and rhs
  %u = rand(n, 1);
  %f = A * u;

  %% --------- Setup Phase: build AMG hierarchy ---------------------------
  % hypre parameters
  setup.print_level  = 0;
  setup.coarsen_type = 10;
  setup.relax_type   = 3;
  setup.relax_sweeps = 1;
  setup.max_level    = 20;
  setup.sol_tol      = 1e-8;
  setup.max_iter     = 1000;
  setup.theta        = 0.3;
  setup.solver_id    = 1;

  % 0: AMG 1: AMG + CG
  % amg is an array of struct that has fields {cf, A, P, v, f}
  % amg(i) for level i of the hierarchy
  % amg(i).{cf, A, P} are computed by hypre (sequential, i.e., NP = 1)
  % cf: C_PT = 1; F_PT = -1
  % amg(i).{v, f} are work space for solve phase

  [amg, y] = hypre_amg(A, b, setup);


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
    post_smoother = pre_smoother;
  end
  % num of relax steps
  mu = 5;

  %% -------- Solve Phase: V - cycle
  fprintf(1, '- - - - - - -  AMG SOLVE PHASE (MATLAB) - - - - - - - - - -\n');

  tol    = setup.sol_tol;
  maxits = setup.max_iter;

  % initial guess
  v = zeros(n, 1);
  r = b - A * v;
  [ProjGrad, activeSet] = ComputeProjGrad(v, -r, lb, ub);
  normr = norm(ProjGrad);
  res(1) = normr;
  corrA(1) = 1;

  iter = 0;

  fprintf(1, 'Cycle    |corr|_A         ProjResidual    |activeSet|\n')
  fprintf(1, '%5d    %.6e     %.6e     %d\n', 0, corrA(1), res(1), numel(activeSet));
  
  while (corrA(iter+1) > tol && iter < maxits)
    iter = iter + 1;
    v_new = Vcycle_MMG(amg, b, v, mu, lb, ub, pre_smoother, post_smoother);
    r = b - A * v;
    [ProjGrad, activeSet] = ComputeProjGrad(v_new, -r, lb, ub);

    normr = norm(ProjGrad);
    res(iter + 1) = normr;
    corrA(iter + 1) = sqrt( (v_new-v)'*A*(v_new-v));
    v = v_new;
    fprintf(1, '%5d    %.6e     %.6e     %d\n', iter, corrA(iter+1), res(iter + 1), numel(activeSet));
  end

  %%
  figure;
  subplot(2, 1, 1); semilogy(res, '--ro'); title('Residue norms');
  subplot(2, 1, 2); semilogy(corrA, '--bx'); title('Correction in Energy norm');

end