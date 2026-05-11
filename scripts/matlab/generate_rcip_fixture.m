%GENERATE_RCIP_FIXTURE Generate RCIP recursion parity fixture.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
[~, outdir] = fixture_context();

pref = [];
pref.k = 4;
cparams = [];
cparams.nchmin = 2;
cparams.ifclosed = false;

[edge1, ~] = chunkerfunc(@(t) chnk.curves.linefunc(t, [0; 0], [1; 0]), cparams, pref);
[edge2, ~] = chunkerfunc(@(t) chnk.curves.linefunc(t, [1; 0], [1; 1]), cparams, pref);
edge1 = sort(edge1);
edge2 = sort(edge2);
chnkrs(1) = edge1;
chnkrs(2) = edge2;

ndim = 1;
nedge = 2;
nsub = 2;
isstart = [true, false];
[Pbc, PWbc, starL, circL, starS, circS, ilist, starL1, circL1] = chnk.rcip.setup(pref.k, ndim, nedge, isstart);
[sbclmat, sbcrmat, lvmat, rvmat, u] = chnk.rcip.shiftedlegbasismats(pref.k);
opts = [];
opts.rcip_savedepth = 2;
fkern = kernel('lap', 'd');
iedgechunks = [1, 2; edge1.nch, 1];
vert0 = [1; 0];

[R, rcipsav] = chnk.rcip.Rcompchunk( ...
    chnkrs, iedgechunks, fkern, ndim, vert0, ...
    Pbc, PWbc, nsub, starL, circL, starS, circS, ilist, starL1, circL1, ...
    sbclmat, sbcrmat, lvmat, rvmat, u, opts);

rcip_fixture = [];
rcip_fixture.edge1 = fixture_pack_chunker(edge1);
rcip_fixture.edge2 = fixture_pack_chunker(edge2);
rcip_fixture.iedgechunks0 = [0, 1; edge1.nch - 1, 0];
rcip_fixture.vert0 = vert0;
rcip_fixture.R = R;
rcip_fixture.saved_R_final = rcipsav.R{end};
rcip_fixture.saved_MAT_last = rcipsav.MAT{end};
rcip_fixture.saved_local_last = fixture_pack_chunker(rcipsav.chnkrlocals{end});
rcip_fixture.sbclmat = sbclmat;
rcip_fixture.sbcrmat = sbcrmat;
rcip_fixture.lvmat = lvmat;
rcip_fixture.rvmat = rvmat;
rcip_fixture.u = u;

save(fullfile(outdir, 'rcip.mat'), 'rcip_fixture');
