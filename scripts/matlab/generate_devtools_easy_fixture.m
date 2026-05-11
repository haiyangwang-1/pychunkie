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

save(fullfile(outdir, 'devtools_easy.mat'), 'devtools_easy', '-v7');


function [d, d2] = local_absconvgauss_der(x, a, b, h)
[~, d, d2] = chnk.spcl.absconvgauss(x, a, b, h);
end
