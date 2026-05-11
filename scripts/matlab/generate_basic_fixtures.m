%GENERATE_BASIC_FIXTURES Generate all committed MATLAB chunkIE fixtures.
%
% Run from the repository root after cloning external/chunkie-matlab:
%
%   matlab -batch "run('scripts/matlab/generate_basic_fixtures.m')"

script_dir = fileparts(mfilename('fullpath'));

run(fullfile(script_dir, 'generate_lege_basic_fixture.m'));
run(fullfile(script_dir, 'generate_chunker_circle_fixture.m'));
run(fullfile(script_dir, 'generate_geometry_core_fixture.m'));
run(fullfile(script_dir, 'generate_lege_extended_fixture.m'));
run(fullfile(script_dir, 'generate_chunker_ops_fixture.m'));
run(fullfile(script_dir, 'generate_kernel_pointinfo_fixture.m'));
run(fullfile(script_dir, 'generate_operator_parity_fixture.m'));
run(fullfile(script_dir, 'generate_devtools_easy_fixture.m'));
