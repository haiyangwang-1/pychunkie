%GENERATE_BASIC_FIXTURES Generate MATLAB chunkIE fixtures for Python tests.
%
% Run from the repository root after cloning external/chunkie-matlab:
%
%   matlab -batch "run('scripts/matlab/generate_basic_fixtures.m')"

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
chunkie_root = fullfile(repo, 'external', 'chunkie-matlab');
addpath(chunkie_root);
run(fullfile(chunkie_root, 'startup.m'));

outdir = fullfile(repo, 'tests', 'golden');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

% Legendre basics
k = 16;
[x, w, u, v] = lege.exps(k);
dmat = lege.dermat(k, u, v);
save(fullfile(outdir, 'lege_basic.mat'), 'k', 'x', 'w', 'u', 'v', 'dmat');

% Circle chunker basics
cparams = [];
cparams.nchmin = 4;
pref = [];
pref.k = 16;
radius = 2.5;
fcurve = @(t) circle_curve(t, radius);
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

function [r, d, d2] = circle_curve(t, radius)
    r = radius * [cos(t(:).'); sin(t(:).')];
    d = radius * [-sin(t(:).'); cos(t(:).')];
    d2 = radius * [-cos(t(:).'); -sin(t(:).')];
end
