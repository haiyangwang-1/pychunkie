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
