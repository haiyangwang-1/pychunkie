%GENERATE_KERNEL_POINTINFO_FIXTURE Generate point-kernel parity fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

srcinfo = fixture_make_ptinfo([0.1, -0.8, 1.25; -0.3, 0.55, 0.9]);
targinfo = fixture_make_ptinfo([-0.45, 0.7; 1.1, -0.65]);
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
stok2d.cpres = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'cpres', stok_coefs);
stok2d.ctrac = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'ctrac', stok_coefs);
stok2d.cgrad = chnk.stok2d.kern(stok_mu, srcinfo, targinfo, 'cgrad', stok_coefs);

elast2d = [];
elast2d.s = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 's');
elast2d.sgrad = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'sgrad');
elast2d.strac = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'strac');
elast2d.d = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'd');
elast2d.dalt = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'dalt');
elast2d.dalttrac = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'dalttrac');
elast2d.daltgrad = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'daltgrad');

greens = [];
[greens.lap.val, greens.lap.grad, greens.lap.hess] = chnk.lap2d.green(srcinfo.r, targinfo.r);
[greens.helm2d.val, greens.helm2d.grad, greens.helm2d.hess] = chnk.helm2d.green(helm_zk, srcinfo.r, targinfo.r);
[greens.helm1d.val, greens.helm1d.grad, greens.helm1d.hess] = chnk.helm1d.green(helm1d_zk, srcinfo.r, targinfo.r);

sweep_uin = [1.0 + 0.25i; -0.5 + 0.1i; 0.75 - 0.2i];
sweep_inds = [1; 3; 5];
sweep_ts = [-1.0; -0.4; 0.0; 0.45; 1.1];
sweep_wts = [0.2; 0.35; 0.5; 0.35; 0.2];
greens.helm1d_sweep.uin = sweep_uin;
greens.helm1d_sweep.inds = sweep_inds;
greens.helm1d_sweep.ts = sweep_ts;
greens.helm1d_sweep.wts = sweep_wts;
greens.helm1d_sweep.out = chnk.helm1d.sweep(sweep_uin, sweep_inds, sweep_ts, sweep_wts, helm1d_zk);

biharm2d = [];
[bh_val, bh_grad, bh_hess] = chnk.flex2d.bhgreen(srcinfo.r, targinfo.r);
biharm2d.green.val = bh_val;
biharm2d.green.grad = bh_grad;
biharm2d.green.hess = bh_hess;
biharm2d.green.lap = bh_hess(:,:,1) + bh_hess(:,:,3);
biharm2d.kern.s = bh_val;
biharm2d.kern.lap = biharm2d.green.lap;
biharm2d.kern.d = -(bh_grad(:,:,1).*srcinfo.n(1,:) + bh_grad(:,:,2).*srcinfo.n(2,:));
biharm2d.kern.sp = bh_grad(:,:,1).*targinfo.n(1,:).' + bh_grad(:,:,2).*targinfo.n(2,:).';
biharm2d.kern.sgrad = zeros(2*size(targinfo.r,2), size(srcinfo.r,2));
biharm2d.kern.sgrad(1:2:end,:) = bh_grad(:,:,1);
biharm2d.kern.sgrad(2:2:end,:) = bh_grad(:,:,2);
biharm2d.kern.shess = zeros(3*size(targinfo.r,2), size(srcinfo.r,2));
biharm2d.kern.shess(1:3:end,:) = bh_hess(:,:,1);
biharm2d.kern.shess(2:3:end,:) = bh_hess(:,:,2);
biharm2d.kern.shess(3:3:end,:) = bh_hess(:,:,3);

kernel_objects = [];
kernel_objects.lap_s = capture_kernel(kernel('lap', 's'), srcinfo, targinfo);
kernel_objects.lap_d = capture_kernel(kernel('lap', 'd'), srcinfo, targinfo);
kernel_objects.lap_sp = capture_kernel(kernel('lap', 'sp'), srcinfo, targinfo);
kernel_objects.lap_stau = capture_kernel(kernel('lap', 'stau'), srcinfo, targinfo);
kernel_objects.lap_sgrad = capture_kernel(kernel('lap', 'sgrad'), srcinfo, targinfo);
kernel_objects.lap_dgrad = capture_kernel(kernel('lap', 'dgrad'), srcinfo, targinfo);
kernel_objects.lap_dp = capture_kernel(kernel('lap', 'dp'), srcinfo, targinfo);
kernel_objects.lap_c = capture_kernel(kernel('lap', 'c', lap_coefs), srcinfo, targinfo);
kernel_objects.lap_cp = capture_kernel(kernel('lap', 'cp', lap_coefs), srcinfo, targinfo);
kernel_objects.lap_cgrad = capture_kernel(kernel('lap', 'cgrad', lap_coefs), srcinfo, targinfo);
kernel_objects.helm2d_s = capture_kernel(kernel('helm', 's', helm_zk), srcinfo, targinfo);
kernel_objects.helm2d_d = capture_kernel(kernel('helm', 'd', helm_zk), srcinfo, targinfo);
kernel_objects.helm2d_sp = capture_kernel(kernel('helm', 'sp', helm_zk), srcinfo, targinfo);
kernel_objects.helm2d_dp = capture_kernel(kernel('helm', 'dp', helm_zk), srcinfo, targinfo);
kernel_objects.helm2d_c = capture_kernel(kernel('helm', 'c', helm_zk, helm_coefs), srcinfo, targinfo);
kernel_objects.helm2d_cp = capture_kernel(kernel('helm', 'cp', helm_zk, helm_coefs), srcinfo, targinfo);
kernel_objects.helm1d_s = capture_kernel(kernel('helm1d', 's', helm1d_zk), srcinfo, targinfo);
kernel_objects.stok_s = capture_kernel(kernel('stok', 's', stok_mu), srcinfo, targinfo);
kernel_objects.stok_spres = capture_kernel(kernel('stok', 'spres', stok_mu), srcinfo, targinfo);
kernel_objects.stok_strac = capture_kernel(kernel('stok', 'strac', stok_mu), srcinfo, targinfo);
kernel_objects.stok_sgrad = capture_kernel(kernel('stok', 'sgrad', stok_mu), srcinfo, targinfo);
kernel_objects.stok_d = capture_kernel(kernel('stok', 'd', stok_mu), srcinfo, targinfo);
kernel_objects.stok_dpres = capture_kernel(kernel('stok', 'dpres', stok_mu), srcinfo, targinfo);
kernel_objects.stok_dtrac = capture_kernel(kernel('stok', 'dtrac', stok_mu), srcinfo, targinfo);
kernel_objects.stok_dgrad = capture_kernel(kernel('stok', 'dgrad', stok_mu), srcinfo, targinfo);
kernel_objects.stok_c = capture_kernel(kernel('stok', 'c', stok_mu, stok_coefs), srcinfo, targinfo);
kernel_objects.stok_cpres = capture_kernel(kernel('stok', 'cpres', stok_mu, stok_coefs), srcinfo, targinfo);
kernel_objects.stok_ctrac = capture_kernel(kernel('stok', 'ctrac', stok_mu, stok_coefs), srcinfo, targinfo);
kernel_objects.stok_cgrad = capture_kernel(kernel('stok', 'cgrad', stok_mu, stok_coefs), srcinfo, targinfo);
kernel_objects.elast_s = capture_kernel(kernel('elast', 's', elast_lam, elast_mu), srcinfo, targinfo);
kernel_objects.elast_sgrad = capture_kernel(kernel('elast', 'sgrad', elast_lam, elast_mu), srcinfo, targinfo);
kernel_objects.elast_strac = capture_kernel(kernel('elast', 'strac', elast_lam, elast_mu), srcinfo, targinfo);
kernel_objects.elast_d = capture_kernel(kernel('elast', 'd', elast_lam, elast_mu), srcinfo, targinfo);
kernel_objects.elast_dalt = capture_kernel(kernel('elast', 'dalt', elast_lam, elast_mu), srcinfo, targinfo);
kernel_objects.elast_dalttrac = capture_kernel(kernel('elast', 'dalttrac', elast_lam, elast_mu), srcinfo, targinfo);
kernel_objects.elast_daltgrad = capture_kernel(kernel('elast', 'daltgrad', elast_lam, elast_mu), srcinfo, targinfo);
kernel_objects.zeros_2_3 = capture_kernel(kernel('zero', 2, 3), srcinfo, targinfo);
kernel_objects.nans_2_3 = capture_kernel(kernel('nan', 2, 3), srcinfo, targinfo);
custom_kernel = kernel(@(s,t) (1.0 + 2.0i)*ones(size(t.r,2), size(s.r,2)) + 0.1*(t.r(1,:).'-s.r(1,:)));
kernel_objects.custom = capture_kernel(custom_kernel, srcinfo, targinfo);

lap_d_obj = kernel('lap', 'd');
lap_s_obj = kernel('lap', 's');
helm_s_obj = kernel('helm', 's', helm_zk);
kernel_algebra = [];
tmp_kernel = 2.0*lap_d_obj + lap_s_obj;
kernel_algebra.add = tmp_kernel.eval(srcinfo, targinfo);
tmp_kernel = lap_d_obj - lap_s_obj;
kernel_algebra.sub = tmp_kernel.eval(srcinfo, targinfo);
tmp_kernel = -lap_d_obj;
kernel_algebra.neg = tmp_kernel.eval(srcinfo, targinfo);
tmp_kernel = helm_s_obj / 2.0;
kernel_algebra.div = tmp_kernel.eval(srcinfo, targinfo);
tmp_kernel = conj(helm_s_obj);
kernel_algebra.conj = tmp_kernel.eval(srcinfo, targinfo);
mixed_kernel = kernel([lap_d_obj, -lap_s_obj; lap_s_obj, kernel('zero')]);
kernel_algebra.interleave = mixed_kernel.eval(srcinfo, targinfo);
kernel_algebra.interleave_meta = kernel_meta(mixed_kernel);

save(fullfile(outdir, 'kernel_pointinfo.mat'), ...
    'srcinfo', 'targinfo', 'lap_coefs', 'helm_zk', 'helm1d_zk', ...
    'helm_coefs', 'helm_all_coefs', 'stok_mu', 'stok_coefs', ...
    'elast_lam', 'elast_mu', 'lap', 'helm2d', 'helm1d', 'stok2d', ...
    'elast2d', 'biharm2d', 'greens', 'kernel_objects', 'kernel_algebra');

function out = capture_kernel(kern, srcinfo, targinfo)
out = [];
out.meta = kernel_meta(kern);
out.eval = kern.eval(srcinfo, targinfo);
end

function meta = kernel_meta(kern)
meta = [];
meta.name = char(kern.name);
meta.type = char(kern.type);
meta.opdims = kern.opdims;
meta.sing = char(kern.sing);
meta.iszero = kern.iszero;
meta.isnan = kern.isnan;
meta.has_eval = isa(kern.eval, 'function_handle');
meta.has_fmm = isa(kern.fmm, 'function_handle');
end
