function [repo, outdir] = fixture_context()
%FIXTURE_CONTEXT Add MATLAB chunkIE paths and return fixture directories.

persistent initialized cached_repo cached_outdir

if isempty(initialized)
    script_dir = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(script_dir));
    chunkie_root = fullfile(repo, 'external', 'chunkie-matlab');
    mingw_bin = 'C:\msys64\mingw64\bin';

    if ispc && exist(mingw_bin, 'dir')
        setenv('PATH', [mingw_bin pathsep getenv('PATH')]);
        setenv('MW_MINGW64_LOC', 'C:\msys64\mingw64');
    end

    addpath(script_dir);
    addpath(chunkie_root);
    run(fullfile(chunkie_root, 'startup.m'));

    outdir = fullfile(repo, 'tests', 'golden');
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    cached_repo = repo;
    cached_outdir = outdir;
    initialized = true;
else
    repo = cached_repo;
    outdir = cached_outdir;
end
end
