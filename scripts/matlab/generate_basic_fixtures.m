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
save(fullfile(outdir, 'chunker_circle.mat'), ...
    'radius', 'ab', 'area_val', 'chunklens', '-struct', 'chnkr');

function [r, d, d2] = circle_curve(t, radius)
    r = radius * [cos(t(:).'); sin(t(:).')];
    d = radius * [-sin(t(:).'); cos(t(:).')];
    d2 = radius * [-cos(t(:).'); -sin(t(:).')];
end
