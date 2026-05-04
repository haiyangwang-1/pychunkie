%GENERATE_BASIC_FIXTURES Generate MATLAB chunkIE fixtures for Python tests.
%
% Run from the repository root after cloning external/chunkie-matlab:
%
%   matlab -batch "run('scripts/matlab/generate_basic_fixtures.m')"

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
chunkie_root = fullfile(repo, 'external', 'chunkie-matlab');
addpath(chunkie_root);
run(fullfile(chunkie_root, 'startup.m'));

outdir = fullfile(repo, 'tests', 'golden');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

% Legendre basics
k = 16;
[x, w, u, v] = lege.exps(k);
dmat = lege.dermat(k, u, v);
save(fullfile(outdir, 'lege_basic.mat'), 'k', 'x', 'w', 'u', 'v', 'dmat');

% Circle chunker basics
cparams = [];
cparams.nchmin = 4;
pref = [];
pref.k = 16;
radius = 2.5;
fcurve = @(t) circle_curve(t, radius);
[chnkr, ab] = chunkerfunc(fcurve, cparams, pref);
area_val = area(chnkr);
chunklens = chunklen(chnkr);
chunker_fields = [];
chunker_fields.r = chnkr.r;
chunker_fields.d = chnkr.d;
chunker_fields.d2 = chnkr.d2;
chunker_fields.n = chnkr.n;
chunker_fields.wts = chnkr.wts;
chunker_fields.adj = chnkr.adj;
chunker_fields.tstor = chnkr.tstor;
chunker_fields.wstor = chnkr.wstor;
chunker_fields.k = chnkr.k;
chunker_fields.nch = chnkr.nch;
chunker_fields.dim = chnkr.dim;
save(fullfile(outdir, 'chunker_circle.mat'), ...
    'radius', 'ab', 'area_val', 'chunklens', 'chunker_fields');

% Broader Legendre helper behavior
k_ext = 7;
xs_ext = [-0.95; -0.4; 0.0; 0.35; 0.9];
coeff_ext = reshape((1:(k_ext*3))/(k_ext*3), k_ext, 3);
[x_ext, w_ext, u_ext, v_ext] = lege.exps(k_ext);
[pol_ext, der_ext] = lege.pols(xs_ext, k_ext - 1);
matrin_ext = lege.matrin(k_ext, xs_ext, u_ext);
intmat_ext = lege.intmat(k_ext, u_ext, v_ext);
exev_ext = lege.exev(xs_ext, coeff_ext);
intpol_true_ext = lege.intpol(coeff_ext, 'true');
intpol_original_ext = lege.intpol(coeff_ext, 'original');
derpol_ext = lege.derpol(coeff_ext);
barywts_ext = lege.barywts(k_ext, x_ext);
save(fullfile(outdir, 'lege_extended.mat'), ...
    'k_ext', 'xs_ext', 'coeff_ext', 'x_ext', 'w_ext', 'u_ext', 'v_ext', ...
    'pol_ext', 'der_ext', 'matrin_ext', 'intmat_ext', 'exev_ext', ...
    'intpol_true_ext', 'intpol_original_ext', 'derpol_ext', 'barywts_ext');

% Chunker transformations and matrix helpers
cparams_ops = [];
cparams_ops.nchmin = 3;
pref_ops = [];
pref_ops.k = 8;
[chnkr_ops, ab_ops] = chunkerfunc(@wobbly_curve, cparams_ops, pref_ops);
mat_ops = [1.15, -0.25; 0.35, 0.8];
chnkr_transformed = mat_ops * move(chnkr_ops, [0.35; -0.2], [0.1; 0.2], 0.45, 1.3);
chunker_ops = [];
chunker_ops.ab = ab_ops;
chunker_ops.mat = mat_ops;
chunker_ops.base = pack_chunker(chnkr_ops);
chunker_ops.transformed = pack_chunker(chnkr_transformed);
chunker_ops.diffmat1 = diffmat(chnkr_ops);
chunker_ops.diffmat2 = diffmat(chnkr_ops, 2);
chunker_ops.intmat = intmat(chnkr_ops);
chunker_ops.onesmat = onesmat(chnkr_ops);
chunker_ops.normonesmat = normonesmat(chnkr_ops);
chunker_ops.centroids = centroids(chnkr_ops);
save(fullfile(outdir, 'chunker_ops.mat'), 'chunker_ops');

% Point-kernel parity on noncoincident source and target info
srcinfo = make_ptinfo([0.1, -0.8, 1.25; -0.3, 0.55, 0.9]);
targinfo = make_ptinfo([-0.45, 0.7; 1.1, -0.65]);
lap_coefs = [1.3; -0.7];
helm_zk = 1.25 + 0.35i;
helm1d_zk = 0.9 + 0.2i;
helm_coefs = [0.8 - 0.1i; -0.45 + 0.3i];
helm_all_coefs = [1.1, -0.4; 0.25, 0.7];
stok_mu = 1.4;
stok_coefs = [0.6; -1.2];
elast_lam = 2.3;
elast_mu = 0.9;

lap = [];
lap.s = chnk.lap2d.kern(srcinfo, targinfo, 's');
lap.d = chnk.lap2d.kern(srcinfo, targinfo, 'd');
lap.sp = chnk.lap2d.kern(srcinfo, targinfo, 'sp');
lap.stau = chnk.lap2d.kern(srcinfo, targinfo, 'stau');
lap.hilb = chnk.lap2d.kern(srcinfo, targinfo, 'hilb');
lap.sgrad = chnk.lap2d.kern(srcinfo, targinfo, 'sgrad');
lap.dgrad = chnk.lap2d.kern(srcinfo, targinfo, 'dgrad');
lap.dp = chnk.lap2d.kern(srcinfo, targinfo, 'dp');
lap.c = chnk.lap2d.kern(srcinfo, targinfo, 'c', lap_coefs);
lap.cp = chnk.lap2d.kern(srcinfo, targinfo, 'cp', lap_coefs);
lap.cgrad = chnk.lap2d.kern(srcinfo, targinfo, 'cgrad', lap_coefs);

helm2d = [];
helm2d.s = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 's');
helm2d.d = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'd');
helm2d.sp = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'sp');
helm2d.stau = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'stau');
helm2d.sgrad = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'sgrad');
helm2d.dgrad = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'dgrad');
helm2d.dp = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'dp');
helm2d.c = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'c', helm_coefs);
helm2d.cp = chnk.helm2d.kern(helm_zk, srcinfo, targinfo, 'cp', helm_coefs);

helm1d = [];
helm1d.s = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 's');
helm1d.d = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'd');
helm1d.sp = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'sp');
helm1d.stau = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'stau');
helm1d.dp = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'dp');
helm1d.c = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'c', helm_coefs);
helm1d.cp = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'cp', helm_coefs);
helm1d.c2trans = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'c2trans', helm_coefs);
helm1d.all = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'all', helm_all_coefs);
helm1d.trans_rep = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'trans_rep', helm_coefs);
helm1d.trans_rep_prime = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'trans_rep_prime', helm_coefs);
helm1d.trans_rep_grad = chnk.helm1d.kern(helm1d_zk, srcinfo, targinfo, 'trans_rep_grad', helm_coefs);

stok2d = [];
stok2d.s = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 's');
stok2d.spres = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'spres');
stok2d.strac = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'strac');
stok2d.d = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'd');
stok2d.dpres = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'dpres');
stok2d.dtrac = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'dtrac');
stok2d.sgrad = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'sgrad');
stok2d.dgrad = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'dgrad');
stok2d.c = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'c', stok_coefs);

elast2d = [];
elast2d.s = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 's');
elast2d.strac = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'strac');
elast2d.d = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'd');
elast2d.dalt = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'dalt');

save(fullfile(outdir, 'kernel_pointinfo.mat'), ...
    'srcinfo', 'targinfo', 'lap_coefs', 'helm_zk', 'helm1d_zk', ...
    'helm_coefs', 'helm_all_coefs', 'stok_mu', 'stok_coefs', ...
    'elast_lam', 'elast_mu', 'lap', 'helm2d', 'helm1d', 'stok2d', 'elast2d');

% Dense operator parity using smooth/native quadrature
cparams_op = [];
cparams_op.nchmin = 4;
pref_op = [];
pref_op.k = 8;
[chnkr_op, ~] = chunkerfunc(@circle_unit, cparams_op, pref_op);
density_scalar = cos(chnkr_op.r(1,:)) + 0.25*sin(2*chnkr_op.r(2,:));
density_stokes = zeros(2*chnkr_op.npt, 1);
density_stokes(1:2:end) = sin(chnkr_op.r(1,:)).';
density_stokes(2:2:end) = cos(chnkr_op.r(2,:)).';
targets = [0.25, 1.7, -1.4; 0.1, -0.45, 0.9];
op_lap_d = @(s,t) chnk.lap2d.kern(s,t,'d');
op_lap_s = @(s,t) chnk.lap2d.kern(s,t,'s');
op_stok_d = @(s,t) chnk.stok2d.kern(stok_mu,s,t,'d');
opts_force_smooth = [];
opts_force_smooth.forcesmooth = true;
operator_parity = [];
operator_parity.chunker = pack_chunker(chnkr_op);
operator_parity.stok_mu = stok_mu;
operator_parity.density_scalar = density_scalar;
operator_parity.density_stokes = density_stokes;
operator_parity.targets = targets;
operator_parity.lap_d_mat = chnk.quadnative.buildmat(chnkr_op, op_lap_d, [1, 1]);
operator_parity.lap_d_apply = operator_parity.lap_d_mat * density_scalar(:);
operator_parity.lap_s_evalmat = chunkerkernevalmat(chnkr_op, op_lap_s, targets, opts_force_smooth);
operator_parity.lap_s_eval = chunkerkerneval(chnkr_op, op_lap_s, density_scalar(:), targets, opts_force_smooth);
operator_parity.stok_d_mat = chnk.quadnative.buildmat(chnkr_op, op_stok_d, [2, 2]);
operator_parity.stok_d_apply = operator_parity.stok_d_mat * density_stokes;
save(fullfile(outdir, 'operator_parity.mat'), 'operator_parity');

function [r, d, d2] = circle_curve(t, radius)
    r = radius * [cos(t(:).'); sin(t(:).')];
    d = radius * [-sin(t(:).'); cos(t(:).')];
    d2 = radius * [-cos(t(:).'); -sin(t(:).')];
end

function [r, d, d2] = circle_unit(t)
    r = [cos(t(:).'); sin(t(:).')];
    d = [-sin(t(:).'); cos(t(:).')];
    d2 = [-cos(t(:).'); -sin(t(:).')];
end

function [r, d, d2] = wobbly_curve(t)
    rad = 1.0 + 0.18*cos(3*t(:).');
    drad = -0.54*sin(3*t(:).');
    d2rad = -1.62*cos(3*t(:).');
    ct = cos(t(:).');
    st = sin(t(:).');
    r = [rad.*ct; rad.*st];
    d = [drad.*ct - rad.*st; drad.*st + rad.*ct];
    d2 = [d2rad.*ct - 2*drad.*st - rad.*ct; ...
          d2rad.*st + 2*drad.*ct - rad.*st];
end

function ptinfo = make_ptinfo(r)
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

function fields = pack_chunker(chnkr)
    fields = [];
    fields.r = chnkr.r;
    fields.d = chnkr.d;
    fields.d2 = chnkr.d2;
    fields.n = chnkr.n;
    fields.wts = chnkr.wts;
    fields.adj = chnkr.adj;
    fields.tstor = chnkr.tstor;
    fields.wstor = chnkr.wstor;
    fields.k = chnkr.k;
    fields.nch = chnkr.nch;
    fields.dim = chnkr.dim;
    fields.area = area(chnkr);
    fields.chunklen = chunklen(chnkr);
end
