%GENERATE_CHUNKER_OPS_FIXTURE Generate chunker operation parity fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

cparams_ops = [];
cparams_ops.nchmin = 3;
pref_ops = [];
pref_ops.k = 8;
[chnkr_ops, ab_ops] = chunkerfunc(@fixture_wobbly_curve, cparams_ops, pref_ops);
mat_ops = [1.15, -0.25; 0.35, 0.8];
chnkr_transformed = mat_ops * move(chnkr_ops, [0.35; -0.2], [0.1; 0.2], 0.45, 1.3);

chunker_ops = [];
chunker_ops.ab = ab_ops;
chunker_ops.mat = mat_ops;
chunker_ops.base = fixture_pack_chunker(chnkr_ops);
chunker_ops.transformed = fixture_pack_chunker(chnkr_transformed);
chunker_ops.diffmat1 = diffmat(chnkr_ops);
chunker_ops.diffmat2 = diffmat(chnkr_ops, 2);
chunker_ops.intmat = intmat(chnkr_ops);
chunker_ops.onesmat = onesmat(chnkr_ops);
chunker_ops.normonesmat = normonesmat(chnkr_ops);
chunker_ops.centroids = centroids(chnkr_ops);

save(fullfile(outdir, 'chunker_ops.mat'), 'chunker_ops');
