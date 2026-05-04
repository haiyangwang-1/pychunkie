%GENERATE_LEGE_EXTENDED_FIXTURE Generate extended Legendre parity fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

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
