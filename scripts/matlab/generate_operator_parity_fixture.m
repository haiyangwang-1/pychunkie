%GENERATE_OPERATOR_PARITY_FIXTURE Generate dense operator parity fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

cparams_op = [];
cparams_op.nchmin = 4;
pref_op = [];
pref_op.k = 8;
[chnkr_op, ~] = chunkerfunc(@fixture_circle_unit, cparams_op, pref_op);
stok_mu = 1.4;
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
operator_parity.chunker = fixture_pack_chunker(chnkr_op);
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
