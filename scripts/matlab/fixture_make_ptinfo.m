function ptinfo = fixture_make_ptinfo(r)
%FIXTURE_MAKE_PTINFO Build deterministic point-info structs for kernels.

d = [0.55 + 0.2*r(2,:); 0.8 - 0.15*r(1,:)];
speed = sqrt(sum(d.^2, 1));
n = [d(2,:)./speed; -d(1,:)./speed];
d2 = [-0.15*r(1,:) + 0.05; 0.2*r(2,:) - 0.03];
ptinfo = [];
ptinfo.r = r;
ptinfo.d = d;
ptinfo.d2 = d2;
ptinfo.n = n;
end
