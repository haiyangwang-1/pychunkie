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
rhohat = (1:size(R, 1)).' / 10;
[rhohatinterp, srcinfo, wts] = chnk.rcip.rhohatInterp(rhohat, rcipsav, nsub);

[T_ip, W_ip] = lege.exps(5);
[IP_ip, IPW_ip] = chnk.rcip.IPinit(T_ip, W_ip);
Pbc_ip = chnk.rcip.Pbcinit(IP_ip, 3, 2);

setup_fixture = [];
setup_fixture.ngl = 4;
setup_fixture.ndim = 2;
setup_fixture.nedge = 3;
setup_fixture.isstart = [true, false, true];
[setup_fixture.Pbc, setup_fixture.PWbc, setup_fixture.starL, ...
    setup_fixture.circL, setup_fixture.starS, setup_fixture.circS, ...
    setup_fixture.ilist, setup_fixture.starL1, setup_fixture.circL1] = ...
    chnk.rcip.setup(setup_fixture.ngl, setup_fixture.ndim, ...
    setup_fixture.nedge, setup_fixture.isstart);

schur_fixture = [];
schur_fixture.ngl = 3;
schur_fixture.ndim = 1;
schur_fixture.nedge = 2;
schur_fixture.isstart = [true, false];
[schur_fixture.Pbc, schur_fixture.PWbc, schur_fixture.starL, ...
    schur_fixture.circL, schur_fixture.starS, schur_fixture.circS] = ...
    chnk.rcip.setup(schur_fixture.ngl, schur_fixture.ndim, ...
    schur_fixture.nedge, schur_fixture.isstart);
rng(1234, 'twister');
nsys = 3 * schur_fixture.ngl * schur_fixture.nedge * schur_fixture.ndim;
nstar = numel(schur_fixture.starL);
ncirc = numel(schur_fixture.circL);
schur_fixture.K = randn(nsys) / 20;
schur_fixture.K(schur_fixture.circL, schur_fixture.circL) = ...
    schur_fixture.K(schur_fixture.circL, schur_fixture.circL) + 5 * eye(ncirc);
schur_fixture.A_input = eye(nstar) + randn(nstar) / 100;
schur_fixture.A_output = chnk.rcip.SchurBana( ...
    schur_fixture.Pbc, schur_fixture.PWbc, schur_fixture.K, ...
    schur_fixture.A_input, schur_fixture.starL, schur_fixture.circL, ...
    schur_fixture.starS, schur_fixture.circS);

corner_fixture = [];
corner_fixture.verts = [0, 1, 1; 0, 0, 1];
corner_fixture.edgesendverts0 = [0, 1; 1, 2];
corner_fixture.edgesendverts1 = corner_fixture.edgesendverts0 + 1;
corner_fixture.k = 4;
corner_fixture.nchmin = 4;
corner_fixture.vertex0 = 1;
corner_fixture.depth = 2;
pref_corner = [];
pref_corner.k = corner_fixture.k;
cparams_corner = [];
cparams_corner.nchmin = corner_fixture.nchmin;
cg_corner = chunkgraph(corner_fixture.verts, corner_fixture.edgesendverts1, ...
    [], cparams_corner, pref_corner);
corner_fixture.original_nch = [cg_corner.echnks.nch];
corner_fixture.vstruc_edges0 = cg_corner.vstruc{corner_fixture.vertex0 + 1}{1} - 1;
corner_fixture.vstruc_signs = cg_corner.vstruc{corner_fixture.vertex0 + 1}{2};
corner_fixture.expected_nch = corner_fixture.original_nch;
corner_fixture.expected_nch(corner_fixture.vstruc_edges0 + 1) = ...
    corner_fixture.expected_nch(corner_fixture.vstruc_edges0 + 1) + corner_fixture.depth;

chunkgraph_fixture = [];
chunkgraph_fixture.verts = [0, 1, 1; 0, 0, 1];
chunkgraph_fixture.edgesendverts0 = [0, 1; 1, 2];
chunkgraph_fixture.vertices0 = 1;
chunkgraph_fixture.edge_indices0 = [0, 1];
chunkgraph_fixture.ndim = ndim;
chunkgraph_fixture.nsub = nsub;
chunkgraph_fixture.rcip_savedepth = opts.rcip_savedepth;
chunkgraph_fixture.R = R;
chunkgraph_fixture.saved_R_final = rcipsav.R{end};
chunkgraph_fixture.saved_MAT_last = rcipsav.MAT{end};

rcip_fixture = [];
rcip_fixture.edge1 = fixture_pack_chunker(edge1);
rcip_fixture.edge2 = fixture_pack_chunker(edge2);
rcip_fixture.iedgechunks0 = [0, 1; edge1.nch - 1, 0];
rcip_fixture.vert0 = vert0;
rcip_fixture.IPinit = [];
rcip_fixture.IPinit.T = T_ip;
rcip_fixture.IPinit.W = W_ip;
rcip_fixture.IPinit.IP = IP_ip;
rcip_fixture.IPinit.IPW = IPW_ip;
rcip_fixture.Pbcinit = [];
rcip_fixture.Pbcinit.IP = IP_ip;
rcip_fixture.Pbcinit.nedge = 3;
rcip_fixture.Pbcinit.ndim = 2;
rcip_fixture.Pbcinit.Pbc = Pbc_ip;
rcip_fixture.setup = setup_fixture;
rcip_fixture.SchurBana = schur_fixture;
rcip_fixture.corner_refine = corner_fixture;
rcip_fixture.chunkgraph_rcip = chunkgraph_fixture;
rcip_fixture.R = R;
rcip_fixture.rhohat = rhohat;
rcip_fixture.rhohatinterp = rhohatinterp;
rcip_fixture.srcinfo = srcinfo;
rcip_fixture.wts = wts;
rcip_fixture.saved_R_final = rcipsav.R{end};
rcip_fixture.saved_MAT_last = rcipsav.MAT{end};
rcip_fixture.saved_local_last = fixture_pack_chunker(rcipsav.chnkrlocals{end});
rcip_fixture.sbclmat = sbclmat;
rcip_fixture.sbcrmat = sbcrmat;
rcip_fixture.lvmat = lvmat;
rcip_fixture.rvmat = rvmat;
rcip_fixture.u = u;

save(fullfile(outdir, 'rcip.mat'), 'rcip_fixture');
