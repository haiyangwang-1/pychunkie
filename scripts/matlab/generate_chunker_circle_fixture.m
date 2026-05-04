%GENERATE_CHUNKER_CIRCLE_FIXTURE Generate circle chunker MATLAB fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

cparams = [];
cparams.nchmin = 4;
pref = [];
pref.k = 16;
radius = 2.5;
fcurve = @(t) fixture_circle_curve(t, radius);
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
