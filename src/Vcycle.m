function v = Vcycle(amg, f, v0, mu, pre_smoother, post_smoother)  % solve equations Ae = r in each level
nlev = length(amg);
n = size(f,2);
amg(1).f = f;
amg(1).v = v0;
%% pre-smoothing
for i=1:nlev-1
    % mu steps pre-relax (e.g., forward G-S)
    % zero inital guess except for the 1st level and the 1st sweep
    for s=1:mu
        if (i > 1 && s == 1)
            amg(i).v = pre_smoother(amg(i).A, amg(i).f);
        else
            amg(i).v = amg(i).v + pre_smoother(amg(i).A, amg(i).f - amg(i).A * amg(i).v);
        end
    end
    % restrict to coarser level
    amg(i+1).f = amg(i).P' * (amg(i).f - amg(i).A * amg(i).v);
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
    for s=1:mu
        amg(i).v = amg(i).v +  post_smoother(amg(i).A, amg(i).f - amg(i).A * amg(i).v);
    end
end
v = amg(1).v;
end