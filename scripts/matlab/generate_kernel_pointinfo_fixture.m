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

elast2d = [];
elast2d.s = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 's');
elast2d.strac = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'strac');
elast2d.d = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'd');
elast2d.dalt = chnk.elast2d.kern(elast_lam, elast_mu, srcinfo, targinfo, 'dalt');

save(fullfile(outdir, 'kernel_pointinfo.mat'), ...
    'srcinfo', 'targinfo', 'lap_coefs', 'helm_zk', 'helm1d_zk', ...
    'helm_coefs', 'helm_all_coefs', 'stok_mu', 'stok_coefs', ...
    'elast_lam', 'elast_mu', 'lap', 'helm2d', 'helm1d', 'stok2d', 'elast2d');
