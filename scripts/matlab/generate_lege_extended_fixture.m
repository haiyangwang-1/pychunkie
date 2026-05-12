%GENERATE_LEGE_EXTENDED_FIXTURE Generate extended Legendre parity fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

k_ext = 7;
xs_ext = [-0.95; -0.4; 0.0; 0.35; 0.9];
coeff_ext = reshape((1:(k_ext*3))/(k_ext*3), k_ext, 3);
[x_ext, w_ext, u_ext, v_ext] = lege.exps(k_ext);
[rts_x_ext, rts_w_ext] = lege.rts(k_ext);
[rts_stab_x_ext, rts_stab_w_ext] = lege.rts_stab(k_ext);
[pol_ext, der_ext] = lege.pols(xs_ext, k_ext - 1);
matrin_ext = lege.matrin(k_ext, xs_ext, u_ext);
intmat_ext = lege.intmat(k_ext, u_ext, v_ext);
exev_ext = lege.exev(xs_ext, coeff_ext);
intpol_true_ext = lege.intpol(coeff_ext, 'true');
intpol_original_ext = lege.intpol(coeff_ext, 'original');
derpol_ext = lege.derpol(coeff_ext);
barywts_ext = lege.barywts(k_ext, x_ext);
bernstein_ntheta_ext = 8;
bernstein_rho_ext = 2.0;
bernstein_ellipse_ext = lege.bernstein_ellipse(bernstein_ntheta_ext, bernstein_rho_ext);
polsum_n_ext = 5;
[polsum_pol_ext, polsum_der_ext, polsum_tot_ext] = lege.polsum(xs_ext, polsum_n_ext);
tayl_x_ext = [-0.4; 0.15; 0.55];
tayl_h_ext = [0.02; -0.03; 0.015];
tayl_n_ext = 6;
tayl_k_ext = 8;
tayl_pol0_ext = zeros(size(tayl_x_ext));
tayl_der0_ext = zeros(size(tayl_x_ext));
tayl_pol_ext = zeros(size(tayl_x_ext));
tayl_der_ext = zeros(size(tayl_x_ext));
for itayl = 1:numel(tayl_x_ext)
    [tayl_pol0_ext(itayl), tayl_der0_ext(itayl)] = lege.pol(tayl_x_ext(itayl), tayl_n_ext);
    [tayl_pol_ext(itayl), tayl_der_ext(itayl)] = lege.tayl( ...
        tayl_pol0_ext(itayl), tayl_der0_ext(itayl), ...
        tayl_x_ext(itayl), tayl_h_ext(itayl), tayl_n_ext, tayl_k_ext);
end
adap_poly_a_ext = -1.0;
adap_poly_b_ext = 2.0;
[adap_poly_val_ext, adap_poly_maxrec_ext, adap_poly_numint_ext, adap_poly_ier_ext] = ...
    lege.adapgauss(@(x) x.^4, adap_poly_a_ext, adap_poly_b_ext);

save(fullfile(outdir, 'lege_extended.mat'), ...
    'k_ext', 'xs_ext', 'coeff_ext', 'x_ext', 'w_ext', 'u_ext', 'v_ext', ...
    'rts_x_ext', 'rts_w_ext', 'rts_stab_x_ext', 'rts_stab_w_ext', ...
    'pol_ext', 'der_ext', 'matrin_ext', 'intmat_ext', 'exev_ext', ...
    'intpol_true_ext', 'intpol_original_ext', 'derpol_ext', 'barywts_ext', ...
    'bernstein_ntheta_ext', 'bernstein_rho_ext', 'bernstein_ellipse_ext', ...
    'polsum_n_ext', 'polsum_pol_ext', 'polsum_der_ext', 'polsum_tot_ext', ...
    'tayl_x_ext', 'tayl_h_ext', 'tayl_n_ext', 'tayl_k_ext', ...
    'tayl_pol0_ext', 'tayl_der0_ext', 'tayl_pol_ext', 'tayl_der_ext', ...
    'adap_poly_a_ext', 'adap_poly_b_ext', 'adap_poly_val_ext', ...
    'adap_poly_maxrec_ext', 'adap_poly_numint_ext', 'adap_poly_ier_ext');
