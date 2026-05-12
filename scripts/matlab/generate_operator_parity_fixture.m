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
rng(137);
compressed_rhs = randn(chnkr_op.npt, 1);
targets = [0.25, 1.7, -1.4; 0.1, -0.45, 0.9];
op_lap_d = @(s,t) chnk.lap2d.kern(s,t,'d');
op_lap_s = @(s,t) chnk.lap2d.kern(s,t,'s');
op_stok_d = @(s,t) chnk.stok2d.kern(stok_mu,s,t,'d');
op_smooth = @(s,t) (t.r(1,:).'-s.r(1,:)).^2 + 0.5*(t.r(2,:).'-s.r(2,:)).^2;
lap_s_kernel = kernel('lap', 's');
opts_force_smooth = [];
opts_force_smooth.forcesmooth = true;
opts_native = [];
opts_native.quad = 'native';
opts_apply = [];
opts_apply.accel = false;
opts_apply.flam = false;
opts_fmm = [];
opts_fmm.forcefmm = true;
opts_fmm.accel = true;
opts_fmm.flam = false;
opts_fmm.eps = 1e-12;
opts_flam = [];
opts_flam.useproxy = false;
opts_flam.occ = 8;
opts_flam.rank_or_tol = 1e-10;
opts_integral = [];
opts_integral.usesmooth = true;
opts_interior = [];
opts_interior.fmm = false;
opts_interior.flam = false;

smooth_kernel = kernel(op_smooth);
smooth_mat = chunkermat(chnkr_op, smooth_kernel, opts_native);
smooth_cormat = sparse(size(smooth_mat, 1), size(smooth_mat, 2));
lap_s_mat = chunkermat(chnkr_op, lap_s_kernel);
lap_s_fmm_apply = chunkermatapply(chnkr_op, lap_s_kernel, compressed_rhs, [], opts_fmm);
flam_factor = chunkerflam(chnkr_op, lap_s_kernel, 1.0, opts_flam);
lap_s_flam_apply = rskelf_mv(flam_factor, compressed_rhs);
lap_s_flam_solve = rskelf_sv(flam_factor, compressed_rhs);

cparams_block = [];
cparams_block.nchmin = 3;
pref_block = [];
pref_block.k = 5;
[block_chnkr1, ~] = chunkerfunc(@fixture_circle_unit, cparams_block, pref_block);
block_chnkr2 = block_chnkr1 + [2.6; 0.25];
block_chnkrs(2,1) = chunker();
block_chnkrs(1,1) = block_chnkr1;
block_chnkrs(2,1) = block_chnkr2;
block_kerns(2,2) = kernel();
block_kerns(1,1) = kernel(@fixture_block_22_kernel);
block_kerns(1,2) = kernel(@fixture_block_21_kernel);
block_kerns(2,1) = kernel(@fixture_block_12_kernel);
block_kerns(2,2) = kernel(@fixture_block_scalar_kernel);
opts_block = [];
opts_block.quad = 'native';
opts_block_flam = opts_flam;
opts_block_flam.quad = 'native';
opts_block_flam.useproxy = false;
opts_block_flam.occ = 1000;
opts_block_flam.rank_or_tol = 1e-12;
block_dense = chunkermat(block_chnkrs, block_kerns, opts_block);
block_dval = 1.25;
opts_block_flam.sp_nonsmooth = spdiags(diag(block_dense), 0, size(block_dense, 1), size(block_dense, 2));
rng(241);
block_rhs = randn(size(block_dense, 2), 1);
block_flam_factor = chunkerflam(block_chnkrs, block_kerns, block_dval, opts_block_flam);
block_flam_apply = rskelf_mv(block_flam_factor, block_rhs);
block_flam_solve = rskelf_sv(block_flam_factor, block_rhs);
interior_points = [-0.2, 1.2, 0.5, -1.4; 0.1, 0.0, 1.1, 0.3];
interior_grid_x = [-1.2, 0.0, 1.2];
interior_grid_y = [-0.8, 0.0, 0.8];

operator_parity = [];
operator_parity.chunker = fixture_pack_chunker(chnkr_op);
operator_parity.pointinfo.r = chnkr_op.r(:,:);
operator_parity.pointinfo.d = chnkr_op.d(:,:);
operator_parity.pointinfo.d2 = chnkr_op.d2(:,:);
operator_parity.pointinfo.n = chnkr_op.n(:,:);
operator_parity.stok_mu = stok_mu;
operator_parity.density_scalar = density_scalar;
operator_parity.density_stokes = density_stokes;
operator_parity.compressed_rhs = compressed_rhs;
operator_parity.targets = targets;
operator_parity.lap_d_mat = chnk.quadnative.buildmat(chnkr_op, op_lap_d, [1, 1]);
operator_parity.lap_d_apply = operator_parity.lap_d_mat * density_scalar(:);
operator_parity.lap_s_mat = lap_s_mat;
operator_parity.lap_s_apply = lap_s_mat * compressed_rhs;
operator_parity.lap_s_fmm_apply = lap_s_fmm_apply;
operator_parity.lap_s_flam_apply = lap_s_flam_apply;
operator_parity.lap_s_flam_solve = lap_s_flam_solve;
operator_parity.block_chunker1 = fixture_pack_chunker(block_chnkr1);
operator_parity.block_chunker2 = fixture_pack_chunker(block_chnkr2);
operator_parity.block_dense = block_dense;
operator_parity.block_dval = block_dval;
operator_parity.block_rhs = block_rhs;
operator_parity.block_flam_apply = block_flam_apply;
operator_parity.block_flam_solve = block_flam_solve;
operator_parity.lap_s_evalmat = chunkerkernevalmat(chnkr_op, op_lap_s, targets, opts_force_smooth);
operator_parity.lap_s_eval = chunkerkerneval(chnkr_op, op_lap_s, density_scalar(:), targets, opts_force_smooth);
operator_parity.stok_d_mat = chnk.quadnative.buildmat(chnkr_op, op_stok_d, [2, 2]);
operator_parity.stok_d_apply = operator_parity.stok_d_mat * density_stokes;
operator_parity.smooth_mat = smooth_mat;
operator_parity.smooth_apply = chunkermatapply(chnkr_op, smooth_kernel, density_scalar(:), smooth_cormat, opts_apply);
operator_parity.zero_mat = chunkermat(chnkr_op, kernel('zero'), opts_native);
operator_parity.integral_values = chunkerintegral(chnkr_op, density_scalar(:), opts_integral);
operator_parity.integral_callable = chunkerintegral(chnkr_op, @(r) r(1,:).^2 + 2*r(2,:).^2, opts_integral);
operator_parity.interior_points = interior_points;
operator_parity.interior_point_flags = chunkerinterior(chnkr_op, interior_points, opts_interior);
operator_parity.interior_grid_x = interior_grid_x;
operator_parity.interior_grid_y = interior_grid_y;
operator_parity.interior_grid_flags = chunkerinterior(chnkr_op, {interior_grid_x, interior_grid_y}, opts_interior);

save(fullfile(outdir, 'operator_parity.mat'), 'operator_parity');

function out = fixture_block_22_kernel(srcinfo, targinfo)
dx = targinfo.r(1,:).' - srcinfo.r(1,:);
dy = targinfo.r(2,:).' - srcinfo.r(2,:);
base = 1.0 + dx.^2 + 0.5*dy.^2;
[nt, ns] = size(base);
out = zeros(2*nt, 2*ns);
out(1:2:end, 1:2:end) = base;
out(2:2:end, 2:2:end) = 2.0 + 0.25*base;
out(1:2:end, 2:2:end) = 0.1*dx;
out(2:2:end, 1:2:end) = -0.2*dy;
end

function out = fixture_block_21_kernel(srcinfo, targinfo)
dx = targinfo.r(1,:).' - srcinfo.r(1,:);
dy = targinfo.r(2,:).' - srcinfo.r(2,:);
base = 0.75 + 0.2*dx - 0.1*dy + 0.05*dx.*dy;
[nt, ns] = size(base);
out = zeros(2*nt, ns);
out(1:2:end, :) = base;
out(2:2:end, :) = 1.25 - 0.15*dx + 0.3*dy.^2;
end

function out = fixture_block_12_kernel(srcinfo, targinfo)
dx = targinfo.r(1,:).' - srcinfo.r(1,:);
dy = targinfo.r(2,:).' - srcinfo.r(2,:);
[nt, ns] = size(dx);
out = zeros(nt, 2*ns);
out(:, 1:2:end) = -0.4 + 0.35*dx.^2 + 0.1*dy;
out(:, 2:2:end) = 0.6 + 0.2*dx - 0.25*dy;
end

function out = fixture_block_scalar_kernel(srcinfo, targinfo)
dx = targinfo.r(1,:).' - srcinfo.r(1,:);
dy = targinfo.r(2,:).' - srcinfo.r(2,:);
out = 1.0 + dx.^2 + 0.5*dy.^2;
end
