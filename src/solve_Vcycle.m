function v = solve_Vcycle(A, b, x, tol, maxits, mu, pre_smoother, post_smoother, amg, opts)
if (opts.print)
    fprintf(1, '- - - - - - -  SOLVE PHASE (MATLAB) - - - - - - - - - - - - - - - - -\n');
end
[m,n] = size(b);
if (isfield(opts, 'init') && size(opts.init,1) == m && size(opts.init,2) == n)
    v = opts.init;
else
    v = zeros(m,n);
end
r = b - A*v;
normr = norm(r);
res(1) = normr;
% tolr = tol * normr;
iter = 0;
if (opts.print)
    err(1) = norm(x-v);
    erra(1) = anorm(A, x-v);
    fprintf(1, 'Iters        ||r||           rate          ||Err||          ||Err||_A\n');
    fprintf(1, '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n');
    fprintf(1, '%5d     %e                  %e      %e\n', ...
        iter, res(iter+1), err(iter+1), erra(iter+1));
end
%
while (normr/res(1) > tol && iter < maxits)
    iter = iter + 1;
    v = Vcycle(amg, b, v, mu, pre_smoother, post_smoother);
    r = b - A*v;
    normr = norm(r);
    res(iter+1) = normr;
    if (opts.print)
        err(iter+1) = norm(x-v);
        erra(iter+1) = anorm(A, x-v);
        fprintf(1, '%5d     %e     %f     %e      %e\n', ...
            iter, res(iter+1), res(iter+1)/res(iter), err(iter+1), erra(iter+1));
    end
end
%
if (opts.print)
    fprintf(1, 'Iter %d, Relres %e [res %e], Err %e \n', ...
        iter, res(end)/res(1), res(end), err(end));
    fprintf(1, 'Average Convergence Factor %f\n', (res(end)/res(1))^(1/iter));
end
end