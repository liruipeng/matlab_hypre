 function T = tridiag(a, b, c, n)
%% function T = tridiag(a, b, c, n)
%% makes the matrix tridiag(a,b,c) of size n x n 
%% 
 sub = diag(ones(n-1,1),-1);
 T = a*eye(n) + b .* sub + c .* sub';
% 
