%% estimating largest eigenvalue using power iteration
function eig = computeMaxEigenvalue(A, x, maxIter)
assert(norm(x) ~= 0, "provide non-zero vector");
x = x / norm(x);
for iter = 1:maxIter
  x = A * x;
  eig = norm(x);
  x = x ./ eig;
end
end