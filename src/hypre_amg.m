function [amg, x] = hypre_amg(A, b, setup)
% NOTE: give hypre A', which is row-wise A
[amg, x] = hypre_amg_setup(A', b, setup);
% add some space in amg for solve
nlev = length(amg);
for i=1:nlev
    ni = size(amg(i).A, 1);
    amg(i).v = zeros(ni,1);
    amg(i).f = zeros(ni,1);
end
end
