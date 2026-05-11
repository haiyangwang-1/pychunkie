%GENERATE_QUADGGQ_FIXTURE Generate special quadrature parity fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

cparams = [];
cparams.nchmin = 6;
pref = [];
pref.k = 8;
[chnkr, ~] = chunkerfunc(@fixture_circle_unit, cparams, pref);

lap_s = @(s,t) chnk.lap2d.kern(s,t,'s');
lap_d = @(s,t) chnk.lap2d.kern(s,t,'d');
lap_sgrad = @(s,t) chnk.lap2d.kern(s,t,'sgrad');
lap_dgrad = @(s,t) chnk.lap2d.kern(s,t,'dgrad');

aux_log = chnk.quadggq.setup(pref.k, 'log');
aux_pv = chnk.quadggq.setup(pref.k, 'pv');
aux_hs = chnk.quadggq.setup(pref.k, 'hs');
aux_removable = chnk.quadggq.setup(pref.k, 'removable');
[removable_xs0, removable_wts0] = chnk.quadggq.getremovablequad(pref.k, 1);

temp = eye(1);
ainterp1kron = kron(aux_log.ainterp1, temp);
ainterps0kron = cell(pref.k, 1);
for inode = 1:pref.k
    ainterps0kron{inode} = kron(aux_log.ainterps0{inode}, temp);
end
source_chunk = 1;
near_target_chunk = chnkr.adj(2, source_chunk);
correction_weights = chnkr.wts;
correction_diag_mask = kron(eye(pref.k), true(1, 1));
correction_diag_mask = correction_diag_mask(:) > 0;

cparams_close = [];
cparams_close.nchmin = 4;
pref_close = [];
pref_close.k = 8;
[close_left, ~] = chunkerfunc(@(t) local_shifted_circle(t, [0; 0], 1.0), cparams_close, pref_close);
[close_right, ~] = chunkerfunc(@(t) local_shifted_circle(t, [2.05; 0], 1.0), cparams_close, pref_close);
close_chnkr = merge([close_left, close_right]);
robust_opts = [];
robust_opts.robust = true;

quadggq = [];
quadggq.chunker = fixture_pack_chunker(chnkr);
quadggq.log_orders = sort(chnk.quadggq.logavail());
quadggq.hqsupp_orders = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20];
quadggq.log_xs1 = aux_log.xs1;
quadggq.log_wts1 = aux_log.wts1;
quadggq.log_xs0 = aux_log.xs0;
quadggq.log_wts0 = aux_log.wts0;
quadggq.removable_xs0 = removable_xs0;
quadggq.removable_wts0 = removable_wts0;
quadggq.setup_removable_xs0 = aux_removable.xs0;
quadggq.setup_removable_wts0 = aux_removable.wts0;
quadggq.pv_xs0 = aux_pv.xs0;
quadggq.pv_wts0 = aux_pv.wts0;
quadggq.hs_xs0 = aux_hs.xs0;
quadggq.hs_wts0 = aux_hs.wts0;
quadggq.native_lap_d_mat = chnk.quadnative.buildmat(chnkr, lap_d, [1, 1]);
quadggq.log_mat = chnk.quadggq.buildmat(chnkr, lap_s, [1, 1], 'log', aux_log);
quadggq.log_mat_skip = chnk.quadggq.buildmat(chnkr, lap_s, [1, 1], 'log', aux_log, [1, 2]);
quadggq.log_td_mat = full(chnk.quadggq.buildmattd(chnkr, lap_s, [1, 1], 'log', aux_log));
quadggq.log_td_mat_skip = full(chnk.quadggq.buildmattd(chnkr, lap_s, [1, 1], 'log', aux_log, [1, 2]));
quadggq.log_td_mat_corrections = full(chnk.quadggq.buildmattd(chnkr, lap_s, [1, 1], 'log', aux_log, [], true));
quadggq.log_diag_chunk = source_chunk;
quadggq.log_near_source_chunk = source_chunk;
quadggq.log_near_target_chunk = near_target_chunk;
quadggq.log_diag_mat = chnk.quadggq.diagbuildmat( ...
    chnkr.r, chnkr.d, chnkr.n, chnkr.d2, chnkr.data, source_chunk, lap_s, [1, 1], ...
    aux_log.xs0, aux_log.wts0, ainterps0kron, aux_log.ainterps0);
quadggq.log_diag_mat_corrections = chnk.quadggq.diagbuildmat( ...
    chnkr.r, chnkr.d, chnkr.n, chnkr.d2, chnkr.data, source_chunk, lap_s, [1, 1], ...
    aux_log.xs0, aux_log.wts0, ainterps0kron, aux_log.ainterps0, true, correction_weights, correction_diag_mask);
quadggq.log_near_mat = chnk.quadggq.nearbuildmat( ...
    chnkr.r, chnkr.d, chnkr.n, chnkr.d2, chnkr.data, near_target_chunk, source_chunk, lap_s, [1, 1], ...
    aux_log.xs1, aux_log.wts1, ainterp1kron, aux_log.ainterp1);
quadggq.log_near_mat_corrections = chnk.quadggq.nearbuildmat( ...
    chnkr.r, chnkr.d, chnkr.n, chnkr.d2, chnkr.data, near_target_chunk, source_chunk, lap_s, [1, 1], ...
    aux_log.xs1, aux_log.wts1, ainterp1kron, aux_log.ainterp1, true, correction_weights);
quadggq.pv_mat = chnk.quadggq.buildmat(chnkr, lap_sgrad, [2, 1], 'pv', aux_pv);
quadggq.hs_mat = chnk.quadggq.buildmat(chnkr, lap_dgrad, [2, 1], 'hs', aux_hs);
quadggq.adap_log_mat = chnk.quadadap.buildmat(chnkr, lap_s, [1, 1], 'log');
quadggq.adap_close_chunker = fixture_pack_chunker(close_chnkr);
quadggq.adap_close_mat = chnk.quadadap.buildmat(close_chnkr, lap_s, [1, 1], 'log');
quadggq.adap_close_robust_mat = chnk.quadadap.buildmat(close_chnkr, lap_s, [1, 1], 'log', robust_opts);

save(fullfile(outdir, 'quadggq.mat'), 'quadggq');

function [r, d, d2] = local_shifted_circle(t, center, radius)
%LOCAL_SHIFTED_CIRCLE Circle curve used for close-interaction quadrature fixtures.

tt = t(:).';
r = center + radius * [cos(tt); sin(tt)];
d = radius * [-sin(tt); cos(tt)];
d2 = radius * [-cos(tt); -sin(tt)];
end
