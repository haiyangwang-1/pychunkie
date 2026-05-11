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
lap_sgrad = @(s,t) chnk.lap2d.kern(s,t,'sgrad');
lap_dgrad = @(s,t) chnk.lap2d.kern(s,t,'dgrad');

aux_log = chnk.quadggq.setup(pref.k, 'log');
aux_pv = chnk.quadggq.setup(pref.k, 'pv');
aux_hs = chnk.quadggq.setup(pref.k, 'hs');

quadggq = [];
quadggq.chunker = fixture_pack_chunker(chnkr);
quadggq.log_xs1 = aux_log.xs1;
quadggq.log_wts1 = aux_log.wts1;
quadggq.log_xs0 = aux_log.xs0;
quadggq.log_wts0 = aux_log.wts0;
quadggq.pv_xs0 = aux_pv.xs0;
quadggq.pv_wts0 = aux_pv.wts0;
quadggq.hs_xs0 = aux_hs.xs0;
quadggq.hs_wts0 = aux_hs.wts0;
quadggq.log_mat = chnk.quadggq.buildmat(chnkr, lap_s, [1, 1], 'log', aux_log);
quadggq.log_mat_skip = chnk.quadggq.buildmat(chnkr, lap_s, [1, 1], 'log', aux_log, [1, 2]);
quadggq.pv_mat = chnk.quadggq.buildmat(chnkr, lap_sgrad, [2, 1], 'pv', aux_pv);
quadggq.hs_mat = chnk.quadggq.buildmat(chnkr, lap_dgrad, [2, 1], 'hs', aux_hs);

save(fullfile(outdir, 'quadggq.mat'), 'quadggq');
