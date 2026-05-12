%GENERATE_DEVTOOLS_EASY_FIXTURE Generate compact fixtures from easy devtools tests.
%
% This script mirrors selected low-dependency cases in
% external/chunkie-matlab/devtools/test without modifying those tests.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[repo, outdir] = fixture_context();

devtools_test_dir = fullfile(repo, 'external', 'chunkie-matlab', 'devtools', 'test');
addpath(devtools_test_dir);

devtools_easy = [];

% absconvgaussTest.m
acg = [];
acg.a = 0.75;
acg.b = -0.75;
acg.m = acg.b/acg.a;
acg.h = acg.b/acg.m/8.0;
acg.offset = 0.0;
acg.ntest = 10;
acg.x = linspace(-acg.a/2, acg.a/2, acg.ntest);
[acg.val, acg.der, acg.der2] = chnk.spcl.absconvgauss(acg.x, acg.m, acg.offset, acg.h);

rng(8675309);
acg.pert = 0.1*acg.b/acg.m;
acg.niter = 6;
acg.errsf = zeros(acg.niter, acg.ntest);
acg.errsg = zeros(acg.niter, acg.ntest);
for i = 1:acg.ntest
    x0 = acg.x(i);
    acg.errsf(:, i) = gradient_check(@(x) chnk.spcl.absconvgauss(x, acg.m, acg.offset, acg.h), ...
        x0, acg.pert, acg.niter, false);
    acg.errsg(:, i) = gradient_check(@(x) local_absconvgauss_der(x, acg.m, acg.offset, acg.h), ...
        x0, acg.pert, acg.niter, false);
end
devtools_easy.absconvgauss = acg;

% legeexpsunitTest.m
leg = [];
rng(8675309);
leg.k = 19;
[leg.x, leg.w, leg.u, leg.v] = lege.exps(leg.k);
leg.dmat = lege.dermat(leg.k, leg.u, leg.v);
leg.imat = lege.intmat(leg.k, leg.u, leg.v);
leg.pv = sin(leg.x);
leg.dpv_true = cos(leg.x);
leg.ipv_true = -cos(leg.x) + cos(-1);
leg.dpv = leg.dmat*leg.pv;
leg.ipv = leg.imat*leg.pv;
leg.cfs = randn(leg.k, 1);
leg.cfsint = lege.intpol(leg.cfs);
leg.cfsint_original = lege.intpol(leg.cfs, 'original');
leg.integral_vals = zeros(leg.k, 1);
for j = 1:leg.k
    leg.integral_vals(j) = integral(@(t) lege.exev(t, leg.cfs), -1, leg.x(j));
end
leg.integral_exev_vals = lege.exev(leg.x, leg.cfsint);
devtools_easy.legeexpsunit = leg;

% arclengthfunTest.m
arc = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-9;
pref = [];
pref.k = 16;
narms = 0;
amp = 0.0;
chnkr = chunkerfunc(@(t) starfish(t, narms, amp), cparams, pref);
[arc.s_single, arc.nchs_single, chnkr] = arclengthfun(chnkr);
arc.chunker_single = fixture_pack_chunker(chnkr);
arc.ts_single = squeeze(atan2(chnkr.r(2,:,:), chnkr.r(1,:,:)));
arc.ts_single(arc.ts_single < 0) = arc.ts_single(arc.ts_single < 0) + 2*pi;

chnkrs(2) = chunker();
chnkrs(1) = chnkr;
arc.rfac = 1.1;
chnkrs(2) = move(chnkr, [0;0], [3;0], 0, arc.rfac);
chnkrtotal = merge(chnkrs);
[arc.s_merged, arc.nchs_merged, chnkrtotal] = arclengthfun(chnkrtotal);
arc.chunker_merged = fixture_pack_chunker(chnkrtotal);
devtools_easy.arclengthfun = arc;

% chunkerarcparamTest.m
cap = [];
cparams = [];
cparams.maxchunklen = 0.3;
cparams.eps = 1.0e-8;
fcurve = @(s) starfish(s, 3, 0.4);
chnkra = chunkerfunc(fcurve, cparams);

cparams = [];
fcurve = @(s) starfish(s, 4, 0, [], pi, 3);
chnkrb = chunkerfunc(fcurve, cparams);

chnkr = merge([chnkra, chnkrb]);
cap.chunker = fixture_pack_chunker(chnkr);
param_data = chnk.arcparam.init(chnkr);
ssa = arclengthfun(chnkra);
ssb = arclengthfun(chnkrb);
cap.lena = sum(chnkra.wts(:));
cap.s_nodes = [ssa, ssb + cap.lena];
[cap.r_nodes, cap.d_nodes, cap.d2_nodes] = chnk.arcparam.eval(cap.s_nodes(:), param_data);
cap.node_residual = norm(chnkr.r(:) - cap.r_nodes(:));

k = chnkr.k;
[xs, ~, us, vs] = lege.exps(k);
cap.a = 0.58;
cap.b = cap.a + 0.2;
cap.sample_s = (cap.b - cap.a) * (xs + 1) / 2 + cap.a;
[cap.sample_r, cap.sample_d, cap.sample_d2] = chnk.arcparam.eval(cap.sample_s, param_data);
dermat = (vs * [lege.derpol(us); zeros(1,k)]).';
dermat = dermat * (2 / (cap.b - cap.a));
cap.der_r_residual = cap.sample_r(:,:) * dermat - cap.sample_d(:,:);
cap.der_d_residual = cap.sample_d(:,:) * dermat - cap.sample_d2(:,:);
cap.orthogonality = sum(cap.sample_d .* cap.sample_d2, 1);

[chnkr2, cap.resampled_eps] = arcresample(chnkr, struct('mv_bdries', 0));
cap.resampled = fixture_pack_chunker(chnkr2);
cap.resampled_area_err = abs(area(chnkr2) - area(chnkr));
cap.resampled_len_err = abs(sum(chnkr2.wts(:)) - sum(chnkr.wts(:)));
cap.resampled_speed_ratio = arclengthdens(chnkr2) ./ (chunklen(chnkr2) / 2).';
devtools_easy.chunkerarcparam = cap;

% chunker_diffintmatTest.m
dimat = [];
chnkr = chunkerfunc(@(t) [cos(t)'; 5*sin(t)']);
D = full(diffmat(chnkr));
C = full(intmat(chnkr));
dx = D * chnkr.r(1,:).';
dy = D * chnkr.r(2,:).';
x = C * dx;
y = C * dy;
dimat.ellipse = fixture_pack_chunker(chnkr);
dimat.ellipse_D = D;
dimat.ellipse_C = C;
dimat.ellipse_dx = dx;
dimat.ellipse_dy = dy;
dimat.ellipse_x_int = x;
dimat.ellipse_y_int = y;
dimat.ellipse_tangent_residual = dx.^2 + dy.^2 - 1;
dimat.ellipse_x_residual = x - x(1) - chnkr.r(1,:).' + chnkr.r(1,1);
dimat.ellipse_y_residual = y - y(1) - chnkr.r(2,:).' + chnkr.r(2,1);

chnkr = chunkerfunc(@(t) [cos(t)'; sin(t)']);
D = full(diffmat(chnkr));
dimat.circle = fixture_pack_chunker(chnkr);
dimat.circle_D = D;
dimat.circle_test_quant = D * chnkr.r(1,:).' + chnkr.r(2,:).';
devtools_easy.chunker_diffintmat = dimat;

% chunker_nearestTest.m
near = [];
rng(1234);
circ = chunkerfunc(@(t) chnk.curves.bymode(t, 1));
near.circle = fixture_pack_chunker(circ);
near.nt = 1000;
near.thetas = -pi + 2*pi*rand(1, near.nt);
near.scal = 0.1 + 2*rand(1, near.nt);
near.targs = [cos(near.thetas); sin(near.thetas)].*near.scal;
near.rn = zeros(2, near.nt);
near.dn = zeros(2, near.nt);
near.d2n = zeros(2, near.nt);
near.dist = zeros(1, near.nt);
near.tn = zeros(1, near.nt);
near.ichn = zeros(1, near.nt);
near.err = zeros(1, near.nt);
for j = 1:near.nt
    [rn, dn, d2n, dist, tn, ichn] = nearest(circ, near.targs(:,j));
    th1 = atan2(near.targs(2,j), near.targs(1,j));
    th2 = atan2(rn(2), rn(1));
    err = min(abs(th1-th2), abs(th1-th2+2*pi));
    err = min(err, abs(th1-th2-2*pi));
    near.rn(:,j) = rn;
    near.dn(:,j) = dn;
    near.d2n(:,j) = d2n;
    near.dist(j) = dist;
    near.tn(j) = tn;
    near.ichn(j) = ichn;
    near.err(j) = err;
end
devtools_easy.chunker_nearest = near;

% chunkerintegralTest.m
cint = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-4;
cint.narms = 5;
cint.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, cint.narms, cint.amp), cparams);
cint.chunker = fixture_pack_chunker(chnkr);
fscal = @(xx) cos(xx(1,:) - 1.0) + sin(xx(2,:) - 0.5);
cint.fvals = fscal(reshape(chnkr.r, 2, chnkr.k*chnkr.nch));
opts = [];
opts.quadgkparams = {'RelTol', 1e-15};
opts.usesmooth = false;
cint.fscal_int1 = chunkerintegral(chnkr, cint.fvals, opts);
opts.usesmooth = true;
cint.fscal_int3 = chunkerintegral(chnkr, cint.fvals, opts);
opts.usesmooth = false;
cint.fscal_int2 = chunkerintegral(chnkr, fscal, opts);
opts.usesmooth = true;
cint.fscal_int4 = chunkerintegral(chnkr, fscal, opts);
cint.relerr12 = abs(cint.fscal_int1 - cint.fscal_int2)/abs(cint.fscal_int2);
cint.relerr32 = abs(cint.fscal_int3 - cint.fscal_int2)/abs(cint.fscal_int2);
cint.relerr42 = abs(cint.fscal_int4 - cint.fscal_int2)/abs(cint.fscal_int2);
devtools_easy.chunkerintegral = cint;

% chunkerfuncuniTest.m
cfu = [];
rng(8675309);
cfu.nch = 10;
cfu.narms = 3;
cfu.amp = 0.5;
chnkr = chunkerfuncuni(@(t) starfish(t, cfu.narms, cfu.amp), cfu.nch);
cfu.starfish = fixture_pack_chunker(chnkr);
[~, ~, info] = sortinfo(chnkr);
cfu.starfish_sort_ier = info.ier;

cfu.modes = randn(11,1);
cfu.modes(1) = 1.1*sum(abs(cfu.modes(2:end)));
cfu.mode_ctr = [1.0; -0.5];
chnkr = chunkerfuncuni(@(t) chnk.curves.bymode(t, cfu.modes, cfu.mode_ctr), cfu.nch);
chnkr = reverse(chnkr);
cfu.bymode_reversed = fixture_pack_chunker(chnkr);

cfu.circle_radius = 5*rand();
cfu.circle_ctr = [1.0; -0.5];
circfun = @(t) cfu.circle_ctr + cfu.circle_radius*[cos(t(:).'); sin(t(:).')];
chnkr = chunkerfuncuni(circfun, cfu.nch);
[~, ~, info] = sortinfo(chnkr);
cfu.circle_sort_ier = info.ier;
cfu.circle = fixture_pack_chunker(chnkr);
cfu.circle_area_error = abs(area(chnkr) - pi*cfu.circle_radius^2);
devtools_easy.chunkerfuncuni = cfu;

% chunkerfuncTest.m
cfunc = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-4;
pref = [];
pref.k = 16;
cfunc.narms = 10;
cfunc.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, cfunc.narms, cfunc.amp), cparams, pref);
cfunc.starfish = fixture_pack_chunker(chnkr);
[~, ~, info] = sortinfo(chnkr);
cfunc.starfish_ier = info.ier;

cparams.nout = 3;
chnkr = chunkerfunc(@(t) starfish(t, cfunc.narms, cfunc.amp), cparams, pref);
cfunc.starfish_nout = fixture_pack_chunker(chnkr);
[~, ~, info] = sortinfo(chnkr);
cfunc.starfish_nout_ier = info.ier;

cfunc.modes = randn(11,1);
cfunc.modes(1) = 1.1 * sum(abs(cfunc.modes(2:end)));
cfunc.mode_ctr = [1.0; -0.5];
chnkr = chunkerfunc(@(t) chnk.curves.bymode(t, cfunc.modes, cfunc.mode_ctr), cparams);
cfunc.bymode = fixture_pack_chunker(chnkr);
[~, ~, info] = sortinfo(chnkr);
cfunc.bymode_ier = info.ier;
chnkr = reverse(chnkr);
cfunc.bymode_reversed = fixture_pack_chunker(chnkr);
[~, ~, info] = sortinfo(chnkr);
cfunc.bymode_reversed_ier = info.ier;

cfunc.circle_radius = 5 * rand();
cfunc.circle_ctr = [1.0; -0.5];
circfun = @(t) cfunc.circle_ctr + cfunc.circle_radius * [cos(t(:).'); sin(t(:).')];
chnkr = chunkerfunc(circfun, cparams);
cfunc.circle = fixture_pack_chunker(chnkr);
[~, ~, info] = sortinfo(chnkr);
cfunc.circle_ier = info.ier;
cfunc.circle_area_error = abs(area(chnkr) - pi * cfunc.circle_radius^2);
chnkr_refined = refine(chnkr, struct('nover', 1));
cfunc.circle_refined = fixture_pack_chunker(chnkr_refined);
cfunc.circle_refined_area_error = abs(area(chnkr_refined) - pi * cfunc.circle_radius^2);

lastwarn('');
cparams_warn = [];
chunkerfunc(@(t) [cos(t(:).'); sin(t(:).'/2)], cparams_warn);
[warnmsg, ~] = lastwarn;
cfunc.closed_warning_seen = ~isempty(warnmsg);

lastwarn('');
cparams_warn = [];
cparams_warn.ta = 0;
cparams_warn.tb = 2*pi - 1e-3;
chunkerfunc(@(t) [cos(t(:).'); sin(t(:).')], cparams_warn);
[warnmsg, ~] = lastwarn;
cfunc.near_closed_warning_seen = ~isempty(warnmsg);

lastwarn('');
cparams_warn.ifclosed = false;
chunkerfunc(@(t) [cos(t(:).'); sin(t(:).')], cparams_warn);
[warnmsg, ~] = lastwarn;
cfunc.open_warning_seen = ~isempty(warnmsg);
devtools_easy.chunkerfunc = cfunc;

% chunkerclassunitTest.m
ccls = [];
ccls.fail_negative_k = false;
try
    pref = [];
    pref.k = -1;
    chunker(pref);
catch
    ccls.fail_negative_k = true;
end
ccls.fail_wrong_nodes = false;
try
    pref = [];
    pref.k = 9;
    [t, w] = lege.exps(8);
    chunker(pref, t, w);
catch
    ccls.fail_wrong_nodes = true;
end
ccls.fail_nchmax = false;
try
    cparams = [];
    pref = [];
    pref.k = 4;
    pref.nchmax = 100;
    chunkerfunc(@(t) starfish(t), cparams, pref);
catch
    ccls.fail_nchmax = true;
end
cparams = [];
cparams.chsmall = 1.0e-14;
cparams.ifclosed = 0;
cparams.tb = 2*pi - 0.01;
cparams.nover = 1;
pref = [];
pref.k = 16;
pref.nchmax = 10000;
chnkr = chunkerfunc(@(t) starfish(t), cparams, pref);
ccls.chunker = fixture_pack_chunker(chnkr);
ccls.adj_ok = true;
for j = 1:chnkr.nch
    i1 = chnkr.adj(1,j);
    i2 = chnkr.adj(2,j);
    if i1 > 0
        ccls.adj_ok = ccls.adj_ok && (chnkr.adj(2,i1) == j);
    end
    if i2 > 0
        ccls.adj_ok = ccls.adj_ok && (chnkr.adj(1,i2) == j);
    end
end
ccls.v = [1;2];
ccls.com1 = chnkr.r(:,:)*chnkr.wts(:)/sum(chnkr.wts(:));
chnkr2 = ccls.v + chnkr;
ccls.plus_left = fixture_pack_chunker(chnkr2);
ccls.com_plus_left = chnkr2.r(:,:)*chnkr2.wts(:)/sum(chnkr2.wts(:));
chnkr2 = chnkr + ccls.v;
ccls.plus_right = fixture_pack_chunker(chnkr2);
ccls.com_plus_right = chnkr2.r(:,:)*chnkr2.wts(:)/sum(chnkr2.wts(:));
ccls.A = [1 2; 2 3];
chnkr2 = ccls.A*chnkr;
ccls.mat_left = fixture_pack_chunker(chnkr2);
ccls.s = 2;
chnkr2 = ccls.s*chnkr;
ccls.scale_left = fixture_pack_chunker(chnkr2);
chnkr2 = chnkr*ccls.s;
ccls.scale_right = fixture_pack_chunker(chnkr2);
ccls.fail_right_matrix = false;
try
    chnkr*ccls.A;
catch
    ccls.fail_right_matrix = true;
end
devtools_easy.chunkerclassunit = ccls;

% chunkerfitTest.m
cfit = [];
rng(0);
cfit.n = 20;
cfit.tt = sort(2*pi*rand(cfit.n,1));
cfit.modes = [2; 0.5; 0.2; 0.7];
cfit.r = chnk.curves.bymode(cfit.tt, cfit.modes);
opts = [];
opts.ifclosed = true;
opts.cparams = [];
opts.cparams.eps = 1.0e-6;
opts.pref = [];
opts.pref.k = 16;
chnkr = chunkerfit(cfit.r, opts);
cfit.closed_ier = checkadjinfo(chnkr);
cfit.closed = fixture_pack_chunker(chnkr);
opts.ifclosed = false;
chnkr = chunkerfit(cfit.r(:,1:10), opts);
cfit.open_ier = checkadjinfo(chnkr);
cfit.open = fixture_pack_chunker(chnkr);
devtools_easy.chunkerfit = cfit;

% tochunkgraphTest.m
tcg = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-9;
pref = [];
pref.k = 16;
tcg.narms = 0;
tcg.amp = 0.0;
chnkr = chunkerfunc(@(t) starfish(t, tcg.narms, tcg.amp), cparams, pref);
tcg.circle = fixture_pack_chunker(chnkr);
cparams.ifclosed = 0;
chnkr2 = chunkerfunc(@(t) local_cos_func(t, pi, 1), cparams, pref);
chnkr2 = move(chnkr2, [0;3]);
tcg.arc = fixture_pack_chunker(chnkr2);
chnkrs(3) = chunker();
chnkrs(1) = chnkr;
tcg.rfac = 1.1;
chnkrs(2) = move(chnkr, [0;0], [3;0], 0, tcg.rfac);
chnkrs(3) = chnkr2;
chnkrtotal = merge(chnkrs);
tcg.total = fixture_pack_chunker(chnkrtotal);
cgrph = tochunkgraph(chnkrtotal);
tcg.graph_verts = cgrph.verts;
tcg.graph_edgesendverts = cgrph.edgesendverts;
tcg.graph_nverts = size(cgrph.verts, 2);
tcg.graph_nedges = length(cgrph.echnks);
tcg.graph_npt = cgrph.npt;
tcg.graph_first_edge = fixture_pack_chunker(cgrph.echnks(1));
tcg.manual_verts = [[0;0], [1;1], [3;0]];
tcg.manual_edge2verts = [[1;2], [3;3]];
cgrph = chunkgraph(tcg.manual_verts, tcg.manual_edge2verts, {chnkr2, chnkr});
tcg.manual_graph_verts = cgrph.verts;
tcg.manual_graph_edgesendverts = cgrph.edgesendverts;
tcg.manual_first_start = cgrph.echnks(1).r(:,1);
tcg.manual_first_end = cgrph.echnks(1).r(:,end);
devtools_easy.tochunkgraph = tcg;

% slicegraphTest.m
slc = [];
verts_out = [[1;1], [1;-1], [-1;-1], [-1;1]];
verts_in = verts_out / 2;
slc.verts = [verts_out, verts_in];
id_vert_out = 1:4;
e2v_out = [id_vert_out; circshift(id_vert_out, 1)];
id_vert_in = 5:8;
e2v_in = [id_vert_in; circshift(id_vert_in, 1)];
slc.edge_2_verts = [e2v_out, e2v_in];
cgrph = chunkgraph(slc.verts, slc.edge_2_verts);
slc.npt = cgrph.npt;
slc.nedges = length(cgrph.echnks);
slc.ichs_mixed = [1, 5:8];
cgrph_slc = slicegraph(cgrph, slc.ichs_mixed);
slc.mixed_r = cgrph_slc.r;
slc.mixed_merge_r = merge(cgrph.echnks(slc.ichs_mixed)).r;
dkern = -2 * kernel('lap', 'd');
slc.full_sysmat = chunkermat(cgrph, dkern);
slc.ichs_inner = 5:8;
cgrph_inner = slicegraph(cgrph, slc.ichs_inner);
slc.inner_sysmat = chunkermat(cgrph_inner, dkern);
slc.idslce = (cgrph.npt - cgrph_inner.npt) + (1:cgrph_inner.npt);
slc.inner_sysmat_from_full = slc.full_sysmat(slc.idslce, slc.idslce);
slc.edgeids_outer_permuted = edgeids(cgrph, [3, 4, 2, 1]);
slc.edgeids_inner = edgeids(cgrph, slc.ichs_inner);
devtools_easy.slicegraph = slc;

% chunkerinteriorTest.m
cint2 = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-9;
pref = [];
pref.k = 16;
cint2.narms = 5;
cint2.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, cint2.narms, cint2.amp), cparams, pref);
cint2.chunker = fixture_pack_chunker(chnkr);
cint2.nt = 10000;
cint2.scal = 2*rand(1, cint2.nt);
cint2.tr = 2*pi*rand(1, cint2.nt);
cint2.targs = bsxfun(@times, starfish(cint2.tr, cint2.narms, cint2.amp), cint2.scal);
opts = [];
opts.flam = false;
opts.fmm = false;
cint2.in = chunkerinterior(chnkr, cint2.targs, opts);
opts = [];
opts.fmm = false;
opts.flam = true;
cint2.in_flam = chunkerinterior(chnkr, cint2.targs, opts);
opts = [];
opts.fmm = true;
opts.flam = false;
cint2.in_fmm = chunkerinterior(chnkr, cint2.targs, opts);
cint2.expected_scal = cint2.scal(:) < 1;

cint2.inner_narms = 3;
cint2.inner_amp = 0.1;
chnkr2 = chunkerfunc(@(t) 0.3*starfish(t, cint2.inner_narms, cint2.inner_amp), cparams, pref);
cint2.inner_chunker = fixture_pack_chunker(chnkr2);
opts = [];
opts.fmm = true;
opts.flam = false;
cint2.in_chunker = chunkerinterior(chnkr, chnkr2, opts);

chnkr_axis = chunkerfunc(@(t) starfish(t), struct('ta', -pi/2, 'tb', pi/2, 'ifclosed', 0));
cint2.axis_chunker = fixture_pack_chunker(chnkr_axis);
cint2.axis_nt = 1000;
cint2.axis_ttarg = -pi/2 + pi*rand(cint2.axis_nt, 1);
cint2.axis_scal = 2*rand(1, cint2.axis_nt);
cint2.axis_targs = starfish(cint2.axis_ttarg).*cint2.axis_scal;
cint2.axis_in = chunkerinterior(chnkr_axis, cint2.axis_targs, struct('axissym', true));
cint2.axis_expected = cint2.axis_scal(:) <= 1;

cint2.stress_amp = 0.25;
cint2.stress_scale = 0.3;
cint2.stress_ctr = [-2; -1.6];
chnkr_int = chunkerfunc(@(t) starfish(t, 3, cint2.stress_amp, cint2.stress_ctr, pi/4, cint2.stress_scale));
chnkr_int = sort(reverse(chnkr_int));
cint2.stress_inner = fixture_pack_chunker(chnkr_int);
a = max(vecnorm(chnkr_int.r(:,:)))*1.01;
cint2.stress_a = a;
chnkr_ext = chunkerfunc(@(t) [a*cos(t(:).'); a*sin(t(:).')]);
cint2.stress_outer = fixture_pack_chunker(chnkr_ext);
chnkr_stress = merge([chnkr_ext, chnkr_int]);
cint2.stress_chunker = fixture_pack_chunker(chnkr_stress);
L = max(abs(chnkr_stress.r), [], "all");
cint2.stress_x = linspace(-L, L, 100);
[xx, yy] = meshgrid(cint2.stress_x, cint2.stress_x);
tt = atan2(yy - cint2.stress_ctr(2), xx - cint2.stress_ctr(1)) + 2*pi;
st = starfish(tt(:), 3, cint2.stress_amp, [0;0], pi/4, cint2.stress_scale);
ss2 = reshape(st(1,:).^2 + st(2,:).^2, size(xx));
cint2.stress_expected = and((xx.^2 + yy.^2) < a^2, ...
    (xx - cint2.stress_ctr(1)).^2 + (yy - cint2.stress_ctr(2)).^2 > ss2);
cint2.stress_in = chunkerinterior(chnkr_stress, {cint2.stress_x, cint2.stress_x});
devtools_easy.chunkerinterior = cint2;

% chunkerpolyTest.m
cpoly = [];
rng(8675309);
cpoly.verts = chnk.demo.barbell(2.0, 2.0, 1.0, 1.0);
cpoly.barb_area = 9;
cpoly.barb_length = 16;
nv = size(cpoly.verts, 2);
cpoly.edgevals = rand(3, nv);
cparams = [];
cparams.widths = 0.1*ones(nv, 1);
cparams.eps = 1.0e-8;
p = [];
p.k = 16;
p.dim = 2;
chnkr = chunkerpoly(cpoly.verts, cparams, p, cpoly.edgevals);
chnkr = chnkr.sort();
cpoly.rounded = fixture_pack_chunker(chnkr);
cpoly.rounded_ier = checkadjinfo(chnkr);

cparams = [];
cparams.rounded = false;
cparams.depth = 8;
chnkr2 = chunkerpoly(cpoly.verts, cparams, p, cpoly.edgevals);
chnkr2 = chnkr2.sort();
cpoly.truepoly = fixture_pack_chunker(chnkr2);
cpoly.truepoly_ier = checkadjinfo(chnkr2);
cpoly.truepoly_area_err = abs(cpoly.barb_area - area(chnkr2))/abs(cpoly.barb_area);
cpoly.truepoly_length_err = abs(cpoly.barb_length - sum(sum(chnkr2.wts)))/abs(cpoly.barb_length);

cpoly.open_verts = randn(2,5);
cparams = [];
cparams.widths = 0.1*ones(size(cpoly.open_verts,2),1);
cparams.autowidths = true;
cparams.autowidthsfac = 0.1;
cparams.ifclosed = 0;
cparams.eps = 1.0e-3;
chnkr3 = chunkerpoly(cpoly.open_verts, cparams, p);
cpoly.open_ier = checkadjinfo(chnkr3);
cpoly.open_nch = chnkr3.nch;
devtools_easy.chunkerpoly = cpoly;

% smootherTest.m
smth = [];
smth.nv = 3;
z = exp(1i*2*pi*(1:smth.nv)/smth.nv);
smth.verts = [real(z); imag(z)];
smth.opts = [];
smth.opts.lam = 10;
[chnkr, smth.err, smth.err_by_pt] = chnk.smoother.smooth(smth.verts, smth.opts);
smth.chunker = fixture_pack_chunker(chnkr);
devtools_easy.smoother = smth;

% flagselfTest.m
fs = [];
rng(8675309);
fs.nsrc_requested = 40000;
xtarg = linspace(0, 1, floor(sqrt(fs.nsrc_requested)));
ytarg = linspace(0, 2, floor(sqrt(fs.nsrc_requested)));
[xxsrc, yysrc] = meshgrid(xtarg, ytarg);
fs.srcs = zeros(2, length(xxsrc(:)));
fs.srcs(1,:) = xxsrc(:);
fs.srcs(2,:) = yysrc(:);
fs.nsrc = size(fs.srcs, 2);
fs.targs_unpermuted = [fs.srcs, [1;2].*rand(2,100)];
fs.P = randperm(size(fs.targs_unpermuted, 2));
fs.targs = fs.targs_unpermuted(:, fs.P) + 1e-15*(2*rand(size(fs.targs_unpermuted))-1);
fs.flagslf = chnk.flagself(fs.srcs, fs.targs);
fs.flagged_count = size(fs.flagslf, 2);
fs.err_count = sum(vecnorm(fs.srcs(:,fs.flagslf(1,:)) - fs.targs(:,fs.flagslf(2,:))) > 1e-10);
devtools_easy.flagself = fs;

% flagnearTest.m
fn = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-6;
pref = [];
pref.k = 16;
fn.narms = 10;
fn.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, fn.narms, fn.amp), cparams, pref);
[~, ~, fn.sortinfo] = sortinfo(chnkr);
fn.chunker = fixture_pack_chunker(chnkr);
fn.fac = 0.7;
fn.nt = 1000;
fn.scal = 2*rand(1, fn.nt);
fn.tr = 2*pi*rand(1, fn.nt);
fn.targs = bsxfun(@times, starfish(fn.tr, fn.narms, fn.amp), fn.scal);
opts = [];
opts.fac = fn.fac;
flag = flagnear(chnkr, fn.targs, opts);
lens = chunklen(chnkr);
flag2 = sparse([], [], [], fn.nt, chnkr.nch);
for i = 1:chnkr.nch
    ris = chnkr.r(:,:,i);
    leni = lens(i);
    for j = 1:chnkr.k
        rj = ris(:,j);
        for l = 1:fn.nt
            dist = sqrt(sum((rj - fn.targs(:,l)).^2, 1));
            if dist < fn.fac*leni
                flag2(l,i) = true;
            end
        end
    end
end
fn.flag = full(flag);
fn.flag_bruteforce = full(flag2);
fn.mismatch_count = nnz(flag2 ~= flag);
devtools_easy.flagnear = fn;

% flagrectTest.m
fr = [];
fr.ngrid = 100;
fr.rho = 1.8;
chnkr = chunkerfunc(@(t) starfish(t));
chnkr = refine(chnkr);
fr.chunker = fixture_pack_chunker(chnkr);
rmin = min(chnkr);
rmax = max(chnkr);
dr = rmax - rmin;
rmin = rmin - dr/2;
rmax = rmax + dr/2;
fr.x = linspace(rmin(1), rmax(1), fr.ngrid);
fr.y = linspace(rmin(2), rmax(2), fr.ngrid);
[xx, yy] = meshgrid(fr.x, fr.y);
fr.targets = [xx(:).'; yy(:).'];
sp = flagnear_rectangle(chnkr, fr.targets);
sp2 = flagnear_rectangle_grid(chnkr, fr.x, fr.y);
fr.flag = full(sp);
fr.flag_grid = full(sp2);
fr.mismatch_count = nnz(sp - sp2);
fr.flagged_count = nnz(sp);
devtools_easy.flagrect = fr;

% helm2d_greenTest.m
h2g = [];
rng(8675309);
h2g.zk = randn() + 1i*randn();
h2g.start_eps = 1.0;
h2g.src = randn(2,1);
h2g.trg = randn(2,1);
h2g.mu = randn(2,1);
h2g.srcn = randn(2,1);
h2g.srcn = h2g.srcn/(norm(h2g.srcn));
[h2g.val, h2g.grad, h2g.hess] = chnk.helm2d.green(h2g.zk, h2g.src, h2g.trg);
h2g.niter = 9;
h2g.errsf = gradient_check(@(x) local_helm2d_green_val(h2g.zk, h2g.src, x), ...
    h2g.trg, h2g.start_eps, h2g.niter, false);
h2g.errsfx = gradient_check(@(x) local_helm2d_green_gradx(h2g.zk, h2g.src, x), ...
    h2g.trg, h2g.start_eps, h2g.niter, false);
h2g.errsfy = gradient_check(@(x) local_helm2d_green_grady(h2g.zk, h2g.src, x), ...
    h2g.trg, h2g.start_eps, h2g.niter, false);
devtools_easy.helm2d_green = h2g;

% kernelopTest.m
kop = [];
rng(8675309);
kop.src = [];
kop.src.r = [[0;0], [0;1]];
kop.src.n = randn(size(kop.src.r));
kop.src.n = kop.src.n./vecnorm(kop.src.n);
kop.targ = [];
kop.targ.r = [[1;1], [-1;0]];
kop.targ.n = randn(size(kop.targ.r));
kop.targ.n = kop.targ.n./vecnorm(kop.targ.n);
skern = kernel('lap', 's');
dkern = kernel('helm', 'd', 1);
kop.a = pi*1i;
fkern1 = kernel([skern; dkern]);
fkern2 = kop.a.*fkern1;
fkern3 = fkern1/kop.a;
nkern = -skern;
ckern1 = skern + dkern;
ckern2 = skern - dkern;
conj_dkern = conj(dkern);
kop.skern = skern.eval(kop.src, kop.targ);
kop.dkern = dkern.eval(kop.src, kop.targ);
kop.fkern1 = fkern1.eval(kop.src, kop.targ);
kop.fkern2 = fkern2.eval(kop.src, kop.targ);
kop.fkern3 = fkern3.eval(kop.src, kop.targ);
kop.nkern = nkern.eval(kop.src, kop.targ);
kop.ckern1 = ckern1.eval(kop.src, kop.targ);
kop.ckern2 = ckern2.eval(kop.src, kop.targ);
kop.conj_dkern = conj_dkern.eval(kop.src, kop.targ);
devtools_easy.kernelop = kop;

% stokes_dtracTest.m
sdtr = [];
rng(8675309);
sdtr.srcinfo = [];
sdtr.srcinfo.r = rand(2,1);
sdtr.srcinfo.n = rand(2,1);
sdtr.targinfo = [];
sdtr.targinfo.r = rand(2,1);
sdtr.targinfo.n = rand(2,1);
sdtr.strengths = rand(2,1);
sdtr.mu = 1.1;
kernt = kernel('stok', 'dtrac', sdtr.mu);
kerng = kernel('stok', 'dgrad', sdtr.mu);
kernp = kernel('stok', 'dpres', sdtr.mu);
sdtr.Kt = kernt.eval(sdtr.srcinfo, sdtr.targinfo)*sdtr.strengths;
sdtr.Kg = kerng.eval(sdtr.srcinfo, sdtr.targinfo)*sdtr.strengths;
sdtr.Kp = kernp.eval(sdtr.srcinfo, sdtr.targinfo)*sdtr.strengths;
du = reshape(sdtr.Kg, [2,2,1]);
dut = permute(du, [2,1,3]);
eu = du + dut;
euxx = squeeze(eu(1,1,:));
euxy = squeeze(eu(1,2,:));
euyy = squeeze(eu(2,2,:));
f = zeros(2,1);
p = sdtr.Kp.';
ntx = sdtr.targinfo.n(1,:).';
nty = sdtr.targinfo.n(2,:).';
f(1:2:end) = -p.*ntx + (euxx.*ntx + euxy.*nty)*sdtr.mu;
f(2:2:end) = -p.*nty + (euxy.*ntx + euyy.*nty)*sdtr.mu;
sdtr.reconstructed = f;
sdtr.residual_norm = norm(f - sdtr.Kt);
devtools_easy.stokes_dtrac = sdtr;

% kernelclassTest.m
kcls = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-12;
pref = [];
pref.k = 16;
kcls.narms = 5;
kcls.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, kcls.narms, kcls.amp), cparams, pref);
kcls.chunker = fixture_pack_chunker(chnkr);
kcls.ns = 10;
ts = 2*pi*rand(kcls.ns, 1);
kcls.sources = 3.0*starfish(ts, kcls.narms, kcls.amp);
kcls.strengths = randn(kcls.ns, 1);
kcls.nt = 100;
ts = 2*pi*rand(kcls.nt, 1);
kcls.targets = starfish(ts, kcls.narms, kcls.amp);
kcls.targets = kcls.targets.*repmat(rand(1, kcls.nt), 2, 1);
kernd = kernel('lap', 'd');
kerns = kernel('lap', 's');
kernsprime = kernel('lap', 'sprime');
srcinfo = [];
srcinfo.r = kcls.sources;
targinfo = [];
targinfo.r = chnkr.r(:,:);
targinfo.d = chnkr.d(:,:);
targinfo.n = chnkr.n(:,:);
kcls.ubdry = kerns.eval(srcinfo, targinfo)*kcls.strengths;
kcls.unbdry = kernsprime.eval(srcinfo, targinfo)*kcls.strengths;
targinfo = [];
targinfo.r = kcls.targets;
kcls.utarg = kerns.eval(srcinfo, targinfo)*kcls.strengths;
opts = [];
kcls.Du = chunkerkerneval(chnkr, kernd, kcls.ubdry, kcls.targets, opts);
kcls.Sun = chunkerkerneval(chnkr, kerns, kcls.unbdry, kcls.targets, opts);
kcls.utarg_identity = kcls.Sun - kcls.Du;
kcls.relerr = norm(kcls.utarg - kcls.utarg_identity, 'fro')/norm(kcls.utarg, 'fro');
opts = [];
opts.forcefmm = true;
kcls.Du_fmm = chunkerkerneval(chnkr, kernd, kcls.ubdry, kcls.targets, opts);
kcls.Sun_fmm = chunkerkerneval(chnkr, kerns, kcls.unbdry, kcls.targets, opts);
kcls.utarg_identity_fmm = kcls.Sun_fmm - kcls.Du_fmm;
kcls.relerr_fmm = norm(kcls.utarg - kcls.utarg_identity_fmm, 'fro')/norm(kcls.utarg, 'fro');
nankern = kernel.nans();
kcls.nankern_isnan = nankern.isnan;
kerntmp = kernd + nankern;
kcls.sum_isnan = kerntmp.isnan;
kerntmp = nan*kerns;
kcls.scaled_isnan = kerntmp.isnan;
devtools_easy.kernelclass = kcls;

% chunkerkerneval_greenlapTest.m
ckgl = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-12;
pref = [];
pref.k = 16;
ckgl.narms = 5;
ckgl.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, ckgl.narms, ckgl.amp), cparams, pref);
ckgl.chunker = fixture_pack_chunker(chnkr);
ckgl.ns = 100;
ts = 2*pi*rand(ckgl.ns, 1);
ckgl.sources = 3.0*starfish(ts, ckgl.narms, ckgl.amp);
ckgl.strengths = randn(ckgl.ns, 1);
ckgl.nt = 300;
ts = 2*pi*rand(ckgl.nt, 1);
ckgl.targets = starfish(ts, ckgl.narms, ckgl.amp);
ckgl.targets = ckgl.targets.*repmat(rand(1, ckgl.nt), 2, 1);
kernd = kernel('lap', 'd');
kerns = kernel('lap', 's');
kernsprime = kernel('lap', 'sprime');
srcinfo = [];
srcinfo.r = ckgl.sources;
targinfo = [];
targinfo.r = chnkr.r(:,:);
targinfo.d = chnkr.d(:,:);
targinfo.n = chnkr.n(:,:);
ckgl.densu = kerns.eval(srcinfo, targinfo)*ckgl.strengths;
ckgl.densun = kernsprime.eval(srcinfo, targinfo)*ckgl.strengths;
targinfo = [];
targinfo.r = ckgl.targets;
ckgl.utarg = kerns.eval(srcinfo, targinfo)*ckgl.strengths;
opts = [];
opts.accel = false;
ckgl.Du_direct = chunkerkerneval(chnkr, kernd, ckgl.densu, ckgl.targets, opts);
ckgl.Sun_direct = chunkerkerneval(chnkr, kerns, ckgl.densun, ckgl.targets, opts);
ckgl.utarg_identity_direct = ckgl.Sun_direct - ckgl.Du_direct;
ckgl.relerr_direct = norm(ckgl.utarg - ckgl.utarg_identity_direct, 'fro')/norm(ckgl.utarg, 'fro');
opts = [];
opts.forcefmm = true;
ckgl.Du_fmm = chunkerkerneval(chnkr, kernd, ckgl.densu, ckgl.targets, opts);
ckgl.Sun_fmm = chunkerkerneval(chnkr, kerns, ckgl.densun, ckgl.targets, opts);
ckgl.utarg_identity_fmm = ckgl.Sun_fmm - ckgl.Du_fmm;
ckgl.relerr_fmm = norm(ckgl.utarg - ckgl.utarg_identity_fmm, 'fro')/norm(ckgl.utarg, 'fro');
opts = [];
opts.flam = true;
ckgl.Du_flam = chunkerkerneval(chnkr, kernd, ckgl.densu, ckgl.targets, opts);
ckgl.Sun_flam = chunkerkerneval(chnkr, kerns, ckgl.densun, ckgl.targets, opts);
ckgl.utarg_identity_flam = ckgl.Sun_flam - ckgl.Du_flam;
ckgl.relerr_flam = norm(ckgl.utarg - ckgl.utarg_identity_flam, 'fro')/norm(ckgl.utarg, 'fro');
ckgl.flam_deferred = false;
devtools_easy.chunkerkerneval_greenlap = ckgl;

% chunkerkernevalmat_greenlapTest.m
ckgmat = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-12;
pref = [];
pref.k = 16;
ckgmat.narms = 5;
ckgmat.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, ckgmat.narms, ckgmat.amp), cparams, pref);
ckgmat.chunker = fixture_pack_chunker(chnkr);
ckgmat.ns = 10;
ts = 2*pi*rand(ckgmat.ns, 1);
ckgmat.sources = 3.0*starfish(ts, ckgmat.narms, ckgmat.amp);
ckgmat.strengths = randn(ckgmat.ns, 1);
ckgmat.nt = 100;
ts = 2*pi*rand(ckgmat.nt, 1);
ckgmat.targets = starfish(ts, ckgmat.narms, ckgmat.amp);
ckgmat.targets = ckgmat.targets.*repmat(rand(1, ckgmat.nt), 2, 1);
kernd = kernel('lap', 'd');
kerns = kernel('lap', 's');
kernsprime = kernel('lap', 'sprime');
srcinfo = [];
srcinfo.r = ckgmat.sources;
targinfo = [];
targinfo.r = chnkr.r(:,:);
targinfo.d = chnkr.d(:,:);
targinfo.n = chnkr.n(:,:);
ckgmat.densu = kerns.eval(srcinfo, targinfo)*ckgmat.strengths;
ckgmat.densun = kernsprime.eval(srcinfo, targinfo)*ckgmat.strengths;
targinfo = [];
targinfo.r = ckgmat.targets;
ckgmat.utarg = kerns.eval(srcinfo, targinfo)*ckgmat.strengths;
opts = [];
ckgmat.Dmat = chunkerkernevalmat(chnkr, kernd, ckgmat.targets, opts);
ckgmat.Smat = chunkerkernevalmat(chnkr, kerns, ckgmat.targets, opts);
ckgmat.Du = ckgmat.Dmat*ckgmat.densu;
ckgmat.Sun = ckgmat.Smat*ckgmat.densun;
ckgmat.utarg_identity = ckgmat.Sun - ckgmat.Du;
ckgmat.relerr = norm(ckgmat.utarg - ckgmat.utarg_identity, 'fro')/norm(ckgmat.utarg, 'fro');
opts = [];
opts.forceadap = true;
ckgmat.Dmat_forceadap = chunkerkernevalmat(chnkr, kernd, ckgmat.targets, opts);
ckgmat.Du_forceadap = ckgmat.Dmat_forceadap*ckgmat.densu;
ckgmat.utarg_identity_forceadap = ckgmat.Sun - ckgmat.Du_forceadap;
ckgmat.relerr_forceadap = norm(ckgmat.utarg - ckgmat.utarg_identity_forceadap, 'fro')/norm(ckgmat.utarg, 'fro');
devtools_easy.chunkerkernevalmat_greenlap = ckgmat;

% chunkerkerneval_greenhelmTest.m
ckgh = [];
rng(8675309);
cparams = [];
cparams.eps = 1.0e-11;
pref = [];
pref.k = 16;
ckgh.narms = 5;
ckgh.amp = 0.5;
chnkr = chunkerfunc(@(t) starfish(t, ckgh.narms, ckgh.amp), cparams, pref);
ckgh.chunker = fixture_pack_chunker(chnkr);
ckgh.ns = 10;
ts = 2*pi*rand(ckgh.ns, 1);
ckgh.sources = 3.0*starfish(ts, ckgh.narms, ckgh.amp);
ckgh.strengths = randn(ckgh.ns, 1);
ckgh.nt = 100;
ts = 2*pi*rand(ckgh.nt, 1);
ckgh.targets = starfish(ts, ckgh.narms, ckgh.amp);
ckgh.targets = ckgh.targets.*repmat(rand(1, ckgh.nt), 2, 1);
ckgh.zk = rand() + 1i*rand();
kernd = kernel('h', 'd', ckgh.zk);
kerns = kernel('h', 's', ckgh.zk);
kernsprime = kernel('h', 'sprime', ckgh.zk);
srcinfo = [];
srcinfo.r = ckgh.sources;
targinfo = [];
targinfo.r = chnkr.r(:,:);
targinfo.d = chnkr.d(:,:);
targinfo.n = chnkr.n(:,:);
ckgh.densu = kerns.eval(srcinfo, targinfo)*ckgh.strengths;
ckgh.densun = kernsprime.eval(srcinfo, targinfo)*ckgh.strengths;
targinfo = [];
targinfo.r = ckgh.targets;
ckgh.utarg = kerns.eval(srcinfo, targinfo)*ckgh.strengths;
opts = [];
ckgh.Du = chunkerkerneval(chnkr, kernd, ckgh.densu, ckgh.targets, opts);
ckgh.Sun = chunkerkerneval(chnkr, kerns, ckgh.densun, ckgh.targets, opts);
ckgh.utarg_identity = ckgh.Sun - ckgh.Du;
ckgh.relerr = norm(ckgh.utarg - ckgh.utarg_identity, 'fro')/norm(ckgh.utarg, 'fro');
devtools_easy.chunkerkerneval_greenhelm = ckgh;

% chunkermat_quadadapTest.m
cqa = [];
rng(8675309);
cqa.zk = randn() + 1i*randn();
cparams = [];
cparams.eps = 1.0e-10;
cparams.nover = 2;
pref = [];
pref.k = 16;
cqa.narms = 3;
cqa.amp = 0.25;
chnkr = chunkerfunc(@(t) starfish(t, cqa.narms, cqa.amp), cparams, pref);
cqa.chunker = fixture_pack_chunker(chnkr);
fkern = @(s,t) chnk.helm2d.kern(cqa.zk, s, t, 'D');
cqa.mat_ggq = chunkermat(chnkr, fkern);
opts = [];
opts.robust = false;
cqa.mat_adap = chnk.quadadap.buildmat(chnkr, fkern, [1 1], 'log', opts);
cqa.relerr = norm(cqa.mat_ggq - cqa.mat_adap, 'fro') / norm(cqa.mat_ggq, 'fro');
devtools_easy.chunkermat_quadadap = cqa;

% datafieldTest.m data-field slices
df = [];
df.hilbert = [];
df.hilbert.kfreq = 5;
cparams = [];
cparams.nover = 2;
chnkr = chunkerfunc(@(t) starfish(t),cparams);
chnkr = chnkr.makedatarows(1);
df.hilbert.L = sum(sum(chnkr.wts));
data_tmp = arclengthfun(chnkr)/df.hilbert.L;
chnkr.data(:) = data_tmp(:)*2*pi;
df.hilbert.chunker = fixture_pack_chunker(chnkr);
df.hilbert.data = chnkr.data(:,:);
hkern = local_H_kernel();
df.hilbert.f1 = sin(2*df.hilbert.kfreq*chnkr.data(:));
H_mat = chunkermat(chnkr,hkern)/df.hilbert.L;
df.hilbert.f2 = H_mat*df.hilbert.f1;
F = chunkerflam(chnkr,hkern,0);
df.hilbert.f2_flam = rskelf_mv(F,df.hilbert.f1);
df.hilbert.err_circle = norm(df.hilbert.f1.^2 + df.hilbert.f2.^2 - 1)/norm(df.hilbert.f1.^2);
df.hilbert.err_flam = norm(df.hilbert.f2-df.hilbert.f2_flam/df.hilbert.L)/norm(df.hilbert.f2);
rng(8675309);
df.nt = 10;
df.srcinfo = [];
df.srcinfo.r = [5;4];
df.targinfo = [];
df.targinfo.r = 0.5*starfish(randn(df.nt,1));
df.v = [2;1];
df.targinfo.data = repmat(df.v,1,df.nt);
chnkr = chunkerfunc(@(t) starfish(t));
df.chunker = fixture_pack_chunker(chnkr);
spkern = kernel('l','sp');
df.unbdry = spkern.eval(df.srcinfo,chnkr);
sysmat = 0.5*eye(chnkr.npt) + onesmat(chnkr) + chunkermat(chnkr,spkern);
df.mu = sysmat\df.unbdry;
kern = local_directional_der_S_kern();
df.deru = chunkerkerneval(chnkr,kern,df.mu,df.targinfo);
opts = [];
opts.forceadap = true;
df.deru_adap = chunkerkerneval(chnkr,kern,df.mu,df.targinfo,opts);
opts = [];
opts.flam = true;
df.deru_flam = chunkerkerneval(chnkr,kern,df.mu,df.targinfo,opts);
sgradkern = kernel('l','sg');
df.gradutrue = sgradkern.eval(df.srcinfo,df.targinfo);
df.derutrue = sum(reshape(df.gradutrue,2,numel(df.gradutrue)/2).*df.v,1);
df.err_direct = norm(df.deru(:)-df.derutrue(:))/norm(df.derutrue(:));
df.err_adap = norm(df.deru_adap(:)-df.deru(:))/norm(df.deru(:));
df.err_flam = norm(df.deru_flam(:)-df.deru(:))/norm(df.deru(:));
devtools_easy.datafield = df;

% FLAM helper geometry
flamh = [];
[flamh.square64_pr, flamh.square64_ptau, flamh.square64_pw, square_pin] = chnk.flam.proxy_square_pts(64);
flamh.square64_inside = square_pin([0.0, 2.0, -1.49, 1.51; 0.0, 0.0, 1.49, 0.0]);
[flamh.circle16_proxy, flamh.circle16_pnorm, flamh.circle16_pw] = chnk.flam.proxy_circ_pts(16);
[flamh.rect_pr, flamh.rect_ptau, flamh.rect_pw] = chnk.flam.proxy_rect_pts([2.0, 3.0], [4, 6]);
flamh.nproxy_lap_s_width = 1.0;
flamh.nproxy_lap_s_nsrc = 30;
flamh.nproxy_lap_s_tol = 1.0e-8;
nproxy_opts = [];
nproxy_opts.nsrc = flamh.nproxy_lap_s_nsrc;
nproxy_opts.rank_or_tol = flamh.nproxy_lap_s_tol;
rng(8675309);
flamh.nproxy_lap_s = chnk.flam.nproxy_square(kernel('lap', 's'), flamh.nproxy_lap_s_width, nproxy_opts);
kbi_cparams = [];
kbi_cparams.nchmin = 4;
kbi_pref = [];
kbi_pref.k = 6;
kbi_chnkr = chunkerfunc(@(t) starfish(t, 0, 0), kbi_cparams, kbi_pref);
flamh.kernbyindex_chunker = fixture_pack_chunker(kbi_chnkr);
flamh.kbi_rows = [1, 3, 8, 12];
flamh.kbi_cols = [2, 5, 9];
flamh.kbi_mat = chnk.flam.kernbyindex(flamh.kbi_rows, flamh.kbi_cols, kbi_chnkr, kernel('lap', 's'), [1, 1]);
kbi_sp = sparse([3, 8], [5, 9], [9.0, -4.0], kbi_chnkr.npt, kbi_chnkr.npt);
flamh.kbi_overwrite = chnk.flam.kernbyindex(flamh.kbi_rows, flamh.kbi_cols, kbi_chnkr, kernel('lap', 's'), [1, 1], kbi_sp);
flamh.kbir_targets = [0.0, 1.4, -0.25; 0.0, 0.2, 1.3];
flamh.kbir_rows = [1, 2, 3];
flamh.kbir_cols = [2, 3, 6];
flamh.kbir_mat = chnk.flam.kernbyindexr(flamh.kbir_rows, flamh.kbir_cols, flamh.kbir_targets, kbi_chnkr, kernel('lap', 's'), [1, 1]);
kbir_sp = sparse([2, 3], [3, 6], [7.0, -3.0], size(flamh.kbir_targets, 2), kbi_chnkr.npt);
flamh.kbir_overwrite = chnk.flam.kernbyindexr(flamh.kbir_rows, flamh.kbir_cols, flamh.kbir_targets, kbi_chnkr, kernel('lap', 's'), [1, 1], kbir_sp);
devtools_easy.flam_helpers = flamh;

save(fullfile(outdir, 'devtools_easy.mat'), 'devtools_easy', '-v7');


function [d, d2] = local_absconvgauss_der(x, a, b, h)
[~, d, d2] = chnk.spcl.absconvgauss(x, a, b, h);
end

function [f, g] = local_helm2d_green_val(zk, src, trg)
[f, g] = chnk.helm2d.green(zk, src, trg);
end

function [f, g] = local_helm2d_green_gradx(zk, src, trg)
[~, g1, h1] = chnk.helm2d.green(zk, src, trg);
f = g1(1);
g = h1(1:2);
end

function [f, g] = local_helm2d_green_grady(zk, src, trg)
[~, g1, h1] = chnk.helm2d.green(zk, src, trg);
f = g1(2);
g = h1(2:3);
end

function kern = local_directional_der_S_kern()
kern = kernel();
kern.opdims = [1,1];
kern.sing = 'pv';
kern.eval = @(s,t) local_directional_der_S(s,t);
end

function submat = local_directional_der_S(s,t)
v = t.data(:,:);
[~,grad] = chnk.lap2d.green(s.r,t.r);
submat = grad(:,:,1).*(v(1,:).') + grad(:,:,2).*(v(2,:).');
end

function kern = local_H_kernel()
kern = kernel();
kern.name = "cotan";
kern.type = "cot";
kern.eval = @(s,t) local_cot_func(s,t);
kern.opdims = [1,1];
kern.sing = 'pv';
end

function val = local_cot_func(s,t)
theta = (t.data(:)) - (s.data(:).');
val = cot(theta/2);
end

function [r, d, d2] = local_cos_func(t, per, amp)
omega = 2*pi/per;
r = [t(:), amp*cos(omega*t(:))].';
d = [ones(length(t),1), -omega*amp*sin(omega*t(:))].';
d2 = [zeros(length(t),1), -omega^2*amp*cos(omega*t(:))].';
end
