function [ProjGrad, activeSet] = ComputeProjGrad(x, grad, lb, ub)
% TODO:: get projected grad, not projected dir
ProjectedConst = x - grad;

idslb = ((x - grad) <= lb);
ProjectedConst(idslb) = lb(idslb);

idsub = ((x - grad) >= ub);
ProjectedConst(idsub) = ub(idsub);

ProjGrad = ProjectedConst - x;
activeSet = sort([find(idslb); find(idsub)]);
end
