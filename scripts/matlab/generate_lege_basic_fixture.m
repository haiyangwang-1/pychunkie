%GENERATE_LEGE_BASIC_FIXTURE Generate basic Legendre MATLAB fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

k = 16;
[x, w, u, v] = lege.exps(k);
dmat = lege.dermat(k, u, v);

save(fullfile(outdir, 'lege_basic.mat'), 'k', 'x', 'w', 'u', 'v', 'dmat');
