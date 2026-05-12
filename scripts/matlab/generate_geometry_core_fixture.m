%GENERATE_GEOMETRY_CORE_FIXTURE Generate MATLAB parity data for I GEOMETRY.
%
% This fixture intentionally avoids chunkerfit and smoothing workflows.

[~, outdir] = fixture_context();

geometry_core = [];

% Top-level domain and curve helpers.
dom = [];
dom.t = [-0.7, 0.0, 0.45, 1.2];
[dom.ellipse_r, dom.ellipse_d, dom.ellipse_d2] = ellipse(dom.t, 2.0, 0.35);
[dom.starfish_r, dom.starfish_d, dom.starfish_d2] = starfish(dom.t, 3, 0.2, [0.1; -0.25], pi/7, 1.4);
[dom.nonflat_r, dom.nonflat_d, dom.nonflat_d2] = nonflatinterface(dom.t, 0.7, 2.0, -0.4, 1.3);
dom.redblue_even = redblue(6);
dom.redblue_odd = redblue(5);
dom.checkcurveparam_dim = checkcurveparam(@(t) ellipse(t, 2.0, 0.35), dom.t, 3);
dom.checkcurveparam_bad_shape = false;
try
    checkcurveparam(@local_bad_shape, dom.t, 1);
catch
    dom.checkcurveparam_bad_shape = true;
end
dom.checkcurveparam_bad_dim = false;
try
    checkcurveparam(@local_bad_dim, dom.t, 2);
catch
    dom.checkcurveparam_bad_dim = true;
end
geometry_core.domain = dom;

curv = [];
curv.t = reshape(dom.t, 2, 2);
[curv.line_r, curv.line_d, curv.line_d2] = chnk.curves.linefunc(curv.t, [0.25; -0.5], [1.75; 0.75]);
[curv.fpara_r, curv.fpara_d, curv.fpara_d2] = chnk.curves.fpara(curv.t, 0.9, -0.15);
[curv.fsine_r, curv.fsine_d, curv.fsine_d2] = chnk.curves.fsine(curv.t, 1.3, 2.4, -0.2);
curv.modes = [1.4; -0.2; 0.35; 0.11; -0.08; 0.03];
curv.center = [0.2; -0.1];
curv.scale = [1.2; 0.8];
[curv.bymode_r, curv.bymode_d, curv.bymode_d2] = chnk.curves.bymode(dom.t, curv.modes, curv.center, curv.scale);
geometry_core.curves = curv;

% Hyper-octree and region helpers.
reg = [];
reg.tree_points = [0.1, 0.9, 0.1, 0.9; 0.1, 0.1, 0.9, 0.9];
reg.tree_extent = [0.0, 1.0; 0.0, 1.0];
tree = hypoct_uni(reg.tree_points, 0.4, Inf, reg.tree_extent);
reg.tree_nlvl = tree.nlvl;
reg.tree_lvp = tree.lvp;
reg.tree_lrt = tree.lrt;
reg.tree_ctr = reshape([tree.nodes.ctr], 2, []);
reg.tree_prnt = zeros(1, numel(tree.nodes));
reg.tree_xi = zeros(numel(tree.nodes), size(reg.tree_points, 2));
reg.tree_chld = zeros(numel(tree.nodes), 4);
reg.tree_nbor = zeros(numel(tree.nodes), 8);
for i = 1:numel(tree.nodes)
    if isempty(tree.nodes(i).prnt)
        reg.tree_prnt(i) = 0;
    else
        reg.tree_prnt(i) = tree.nodes(i).prnt;
    end
    reg.tree_xi(i, 1:numel(tree.nodes(i).xi)) = tree.nodes(i).xi;
    reg.tree_chld(i, 1:numel(tree.nodes(i).chld)) = tree.nodes(i).chld;
    reg.tree_nbor(i, 1:numel(tree.nodes(i).nbor)) = tree.nodes(i).nbor;
end

reg.verts = [
    0.0, 2.0, 2.0, 0.0, 0.75, 1.25, 1.25, 0.75;
    0.0, 0.0, 2.0, 2.0, 0.75, 0.75, 1.25, 1.25
];
reg.edges = [
    1, 2, 3, 4, 5, 6, 7, 8;
    2, 3, 4, 1, 6, 7, 8, 5
];
reg.pref = [];
reg.pref.k = 6;
reg.cparams = [];
reg.cparams.nchmin = 4;
cgrph_regions = chunkgraph(reg.verts, reg.edges, [], reg.cparams, reg.pref);
reg.outer = {{[1, 2, 3, 4]}};
reg.inner = {{[5, 6, 7, 8]}};
rgn_outer = {[], {[1, 2, 3, 4]}};
rgn_inner = {{[5, 6, 7, 8]}};
reg.point_inside = pointinregion(cgrph_regions, reg.outer{1}, [1.0; 1.0]);
reg.point_outside = pointinregion(cgrph_regions, reg.outer{1}, [3.0; 1.0]);
reg.region_inside = regioninside(cgrph_regions, rgn_outer, rgn_inner);
reg.merged = mergeregions(cgrph_regions, rgn_outer, rgn_inner);
reg.merged_region_count = numel(reg.merged);
reg.merged_loop_count = cellfun(@numel, reg.merged);
reg.merged_first_inner_edge_count = numel(reg.merged{2}{1});
reg.merged_second_inner_edge_count = numel(reg.merged{2}{2});
geometry_core.regions = reg;

% Chunker core helpers.
chn = [];
chn.pref = [];
chn.pref.k = 10;
chn.pref.nchmax = 80;
chn.cparams = [];
chn.cparams.eps = 1.0e-3;
chn.cparams.nchmin = 4;
chn.cparams.ifrefine = false;
[base, chn.ab] = chunkerfunc(@fixture_wobbly_curve, chn.cparams, chn.pref);
chn.base = fixture_pack_chunker(base);
chn.weights = weights(base);
chn.normals = normals(base);
chn.tangents = tangents(base);
chn.arclengthdens = arclengthdens(base);
chn.signed_curvature = signed_curvature(base);
chn.arclengthfun = arclengthfun(base);
chn.uvals = sin(reshape(base.r(1,:,:), base.k, base.nch)) + 0.25*cos(reshape(base.r(2,:,:), base.k, base.nch));
chn.arclengthder = arclengthder(base, chn.uvals);
[chn.chunkends_r, chn.chunkends_tau] = chunkends(base, [1, base.nch]);
chn.min = min(base);
chn.max = max(base);

chn.flag_targets = [
    -1.2, -0.4, 0.2, 0.95, 1.4, 0.0;
     0.1,  0.9, 0.0, 0.25, 0.0, -1.3
];
chn.flag_fac = 0.85;
flag_opts = [];
flag_opts.fac = chn.flag_fac;
chn.flagnear = full(flagnear(base, chn.flag_targets, flag_opts));

chn.rect_x = linspace(-1.4, 1.4, 8);
chn.rect_y = linspace(-1.2, 1.2, 7);
[rect_xx, rect_yy] = meshgrid(chn.rect_x, chn.rect_y);
chn.rect_targets = [rect_xx(:).'; rect_yy(:).'];
chn.rect_rho = 1.35;
rect_opts = [];
rect_opts.rho = chn.rect_rho;
chn.flagnear_rectangle = full(flagnear_rectangle(base, chn.rect_targets, rect_opts));
chn.flagnear_rectangle_grid = full(flagnear_rectangle_grid(base, chn.rect_x, chn.rect_y, rect_opts));

chn.nearest_targets = [
    1.25, -0.6, 0.1, 0.35;
    0.15,  1.0, -1.1, 0.0
];
chn.nearest_r = zeros(base.dim, size(chn.nearest_targets, 2));
chn.nearest_d = zeros(base.dim, size(chn.nearest_targets, 2));
chn.nearest_d2 = zeros(base.dim, size(chn.nearest_targets, 2));
chn.nearest_dist = zeros(1, size(chn.nearest_targets, 2));
chn.nearest_t = zeros(1, size(chn.nearest_targets, 2));
chn.nearest_ich = zeros(1, size(chn.nearest_targets, 2));
for iref = 1:size(chn.nearest_targets, 2)
    [chn.nearest_r(:,iref), chn.nearest_d(:,iref), chn.nearest_d2(:,iref), ...
        chn.nearest_dist(iref), chn.nearest_t(iref), chn.nearest_ich(iref)] = ...
        nearest(base, chn.nearest_targets(:,iref));
end

dirty = base;
dirty.n = zeros(size(base.n));
dirty.wts = zeros(size(base.wts));
dirty.n = normals(dirty);
dirty.wts = weights(dirty);
chn.recomputed = fixture_pack_chunker(dirty);

chn.translation_vector = [0.35; -0.45];
chn.translated_left = fixture_pack_chunker(chn.translation_vector + base);
chn.translated_right = fixture_pack_chunker(base + chn.translation_vector);

cfu_params = [];
cfu_params.ta = -0.25;
cfu_params.tb = 2*pi - 0.25;
cfu_params.ifclosed = true;
cfu_pref = [];
cfu_pref.k = 12;
cfu_nch = 5;
cfu_curve = @(t) starfish(t, 4, 0.15, [0.05; -0.1], 0.2, 0.9);
chn.chunkerfuncuni_nch = cfu_nch;
chn.chunkerfuncuni_cparams = cfu_params;
chn.chunkerfuncuni_pref = cfu_pref;
chn.chunkerfuncuni = fixture_pack_chunker(chunkerfuncuni(cfu_curve, cfu_nch, cfu_params, cfu_pref));

[chn.sort_inds, chn.sort_adjs, chn.sort_info] = sortinfo(base);
sorted_base = sort(base);
chn.sorted = fixture_pack_chunker(sorted_base);

data_base = base.makedatarows(2);
data_base.data(1,:,:) = reshape(base.r(1,:,:), 1, base.k, base.nch) + 2.0*reshape(base.r(2,:,:), 1, base.k, base.nch);
data_base.data(2,:,:) = reshape(base.tstor.^9, 1, base.k, 1) .* ones(1, 1, base.nch);
chn.data = data_base.data;
data_opts = [];
data_opts.idata = [1, 2];
data_opts.ncoeff = 3;
data_opts.tol = 1.0e-8;
chn.datares = datares(data_base, data_opts);

split_base = base;
[split_x, split_w, split_u] = lege.exps(base.k);
split_base = split(split_base, 2, [], split_x, split_w, split_u, 't');
split_base.n = normals(split_base);
split_base.wts = weights(split_base);
chn.split_param = fixture_pack_chunker(split_base);

ref_opts = [];
ref_opts.splitchunks = [2];
ref_opts.lvlr = 'n';
ref_opts.stype = 't';
ref_opts.nover = 1;
refined = refine(base, ref_opts);
refined.n = normals(refined);
refined.wts = weights(refined);
chn.refined = fixture_pack_chunker(refined);

sigma = zeros(2, base.k, base.nch);
sigma(1,:,:) = base.r(1,:,:);
sigma(2,:,:) = base.r(2,:,:);
[upsampled, sigmaup] = upsample(data_base, 14, sigma);
chn.upsampled = fixture_pack_chunker(upsampled);
chn.upsampled_data = upsampled.data;
chn.sigma = sigma;
chn.sigmaup = sigmaup;

arc_opts = [];
[arcresampled, arc_eps] = arcresample(base, arc_opts);
chn.arcresampled = fixture_pack_chunker(arcresampled);
chn.arcresample_eps = arc_eps;

rotated = rotate(base, 0.37, [0.2; -0.1], [-0.3; 0.4]);
reflected = reflect(base, -0.2, [0.1; 0.2], [0.25; -0.35]);
reversed_base = reverse(base);
chn.rotated = fixture_pack_chunker(rotated);
chn.reflected = fixture_pack_chunker(reflected);
chn.reversed = fixture_pack_chunker(reversed_base);

points_from_r = chunkerpoints(base.r, struct('ifclosed', true));
src = [];
src.r = base.r;
src.d = 2.0*base.d;
src.d2 = 3.0*base.d2;
points_explicit = chunkerpoints(src, struct('ifclosed', true));
chn.points_from_r = fixture_pack_chunker(points_from_r);
chn.points_explicit = fixture_pack_chunker(points_explicit);

ccirc = [];
ccirc.nchmin = 4;
ccirc.ifrefine = false;
ccirc.eps = 1.0e-3;
pcirc = [];
pcirc.k = 8;
[circ1, ~] = chunkerfunc(@(t) local_circle(t, [0; 0], 0.5), ccirc, pcirc);
[circ2, ~] = chunkerfunc(@(t) local_circle(t, [1.75; -0.25], 0.3), ccirc, pcirc);
circ1 = circ1.makedatarows(1);
circ1.data(1,:,:) = reshape(circ1.r(1,:,:), 1, circ1.k, circ1.nch);
circ2 = circ2.makedatarows(2);
circ2.data(1,:,:) = 5.0;
circ2.data(2,:,:) = reshape(circ2.r(2,:,:), 1, circ2.k, circ2.nch);
merged_circles = merge([circ1, circ2]);
chn.circ1 = fixture_pack_chunker(circ1);
chn.circ1_data = circ1.data;
chn.circ2 = fixture_pack_chunker(circ2);
chn.circ2_data = circ2.data;
chn.merged_circles = fixture_pack_chunker(merged_circles);
chn.merged_circles_data = merged_circles.data;
geometry_core.chunker = chn;

% +chnk geometry helpers.
geo = [];
geo.ptinfo = [];
geo.ptinfo.r = reshape(base.r, 2, base.npt);
geo.ptinfo.d = reshape(base.d, 2, base.npt);
geo.ptinfo.d2 = reshape(base.d2, 2, base.npt);
geo.perp = chnk.perp(geo.ptinfo.d);
geo.normal2d = chnk.normal2d(geo.ptinfo);
geo.curvature2d = chnk.curvature2d(geo.ptinfo);
geo.targets = [0.2, 1.2, -0.4; 0.1, -0.3, 0.7];
[near_x, ~, near_u] = lege.exps(base.k);
[geo.near_t, geo.near_r, geo.near_d, geo.near_d2, geo.near_dist2] = chnk.chunk_nearparam(base.r(:,:,2), geo.targets, [], near_x, near_u);
geo.flagself_src = [0, 1, 2, 3; 0, 1, 2, 3];
geo.flagself_targ = [2, 0, 3, 5; 2, 0, 3, 5];
geo.flagself_pairs = chnk.flagself(geo.flagself_src, geo.flagself_targ);
geometry_core.geometry = geo;

% +chnk arc-length parameterization helpers.
ap = [];
ap.full = chnk.arcparam.init(base);
ap.eval_s = linspace(0.05, sum(ap.full.plen) - 0.05, 9).';
[ap.eval_r, ap.eval_d, ap.eval_d2] = chnk.arcparam.eval(ap.eval_s, ap.full);
ap.node_s = reshape(chn.arclengthfun, [], 1);
[ap.node_r, ap.node_d, ap.node_d2] = chnk.arcparam.eval(ap.node_s, ap.full);
ap.selected_ich = [2, 4];
ap.selected = chnk.arcparam.init(base, ap.selected_ich);
ap.selected_eval_s = linspace(0.03, sum(ap.selected.plen) - 0.03, 6).';
[ap.selected_eval_r, ap.selected_eval_d, ap.selected_eval_d2] = chnk.arcparam.eval(ap.selected_eval_s, ap.selected);
geometry_core.arcparam = ap;

% Chunkgraph helpers.
cgfx = [];
cgfx.verts = [0, 1, 1, 0; 0, 0, 1, 1];
cgfx.edges = [1, 2, 3, 4; 2, 3, 4, 1];
cgfx.pref = [];
cgfx.pref.k = 6;
cgfx.cparams = [];
cgfx.cparams.nchmin = 4;
cgrph = chunkgraph(cgfx.verts, cgfx.edges, [], cgfx.cparams, cgfx.pref);
cgfx.copy_delta = 4.25;
copy_source = cgrph;
copy_mutated = copy_source;
copy_mutated.verts(1,1) = copy_mutated.verts(1,1) + cgfx.copy_delta;
copy_mutated.echnks(1).rstor(1,1,1) = copy_mutated.echnks(1).rstor(1,1,1) - cgfx.copy_delta;
cgfx.copy_source_vert = copy_source.verts(1,1);
cgfx.copy_mutated_vert = copy_mutated.verts(1,1);
cgfx.copy_source_edge_node = copy_source.echnks(1).rstor(1,1,1);
cgfx.copy_mutated_edge_node = copy_mutated.echnks(1).rstor(1,1,1);
cgfx.edgesendverts = cgrph.edgesendverts;
cgfx.v2emat = full(cgrph.v2emat);
cgfx.npt = cgrph.npt;
cgfx.r = cgrph.r;
cgfx.d = cgrph.d;
cgfx.d2 = cgrph.d2;
cgfx.n = cgrph.n;
cgfx.wts = cgrph.wts;
cgfx.adj = cgrph.adj;
cgfx.min = min(cgrph);
cgfx.max = max(cgrph);
cgfx.onesmat = onesmat(cgrph);
cgfx.normonesmat = normonesmat(cgrph);
cgfx.sourceinfo_r = cgrph.sourceinfo.r;
cgfx.sourceinfo_n = cgrph.sourceinfo.n;
cgfx.sourceinfo_d = cgrph.sourceinfo.d;
cgfx.sourceinfo_d2 = cgrph.sourceinfo.d2;
cgfx.sourceinfo_w = cgrph.sourceinfo.w;
cgfx.edgeids = edgeids(cgrph, [1, 3]);
cgfx.sliced = slicegraph(cgrph, [1, 3]);
cgfx.sliced_edgesendverts = cgfx.sliced.edgesendverts;
cgfx.sliced_v2emat = full(cgfx.sliced.v2emat);
cgfx.sliced_npt = cgfx.sliced.npt;
cgfx.procverts_edges = zeros(2, size(cgfx.verts, 2));
cgfx.procverts_signs = zeros(2, size(cgfx.verts, 2));
cgfx.procverts_counts = zeros(1, size(cgfx.verts, 2));
for iv = 1:numel(cgrph.vstruc)
    edges_here = cgrph.vstruc{iv}{1};
    signs_here = cgrph.vstruc{iv}{2};
    cgfx.procverts_counts(iv) = numel(edges_here);
    cgfx.procverts_edges(1:numel(edges_here), iv) = edges_here;
    cgfx.procverts_signs(1:numel(signs_here), iv) = signs_here;
end
cgfx.region_count = numel(cgrph.regions);
cgfx.region_loop_count = cellfun(@numel, cgrph.regions);
cgfx.region_first_loop = cgrph.regions{1}{1};
cgfx.region_second_loop = cgrph.regions{2}{1};

graph_ref_opts = [];
graph_ref_opts.nover = 1;
graph_ref_opts.lvlr = 'n';
cgfx.refine_opts = graph_ref_opts;
cgfx.refined = local_pack_chunkgraph(refine(cgrph, graph_ref_opts));

cgfx.flag_targets = [
    -0.2, 0.5, 1.2, 0.5, 0.5;
     0.5, 0.5, 0.5, -0.2, 1.2
];
cgfx.flag_fac = 0.8;
graph_flag_opts = [];
graph_flag_opts.fac = cgfx.flag_fac;
cgfx.flagnear = full(flagnear(cgrph, cgfx.flag_targets, graph_flag_opts));
cgfx.rect_x = linspace(-0.25, 1.25, 6);
cgfx.rect_y = linspace(-0.25, 1.25, 5);
[cg_rect_xx, cg_rect_yy] = meshgrid(cgfx.rect_x, cgfx.rect_y);
cgfx.rect_targets = [cg_rect_xx(:).'; cg_rect_yy(:).'];
cgfx.rect_rho = 1.25;
graph_rect_opts = [];
graph_rect_opts.rho = cgfx.rect_rho;
cgfx.flagnear_rectangle = full(flagnear_rectangle(cgrph, cgfx.rect_targets, graph_rect_opts));
cgfx.flagnear_rectangle_grid = full(flagnear_rectangle_grid(cgrph, cgfx.rect_x, cgfx.rect_y, graph_rect_opts));

cgfx.translation_vector = [0.4; -0.25];
cgfx.transform_matrix = [1.1, 0.2; -0.3, 0.9];
cgfx.scale = 1.35;
translated = cgfx.translation_vector + cgrph;
translated_right = cgrph + cgfx.translation_vector;
transformed = cgfx.transform_matrix * cgrph;
scaled_left = cgfx.scale * cgrph;
scaled_right = cgrph * cgfx.scale;
rotated_graph = rotate(cgrph, 0.31, [0.5; 0.5], [0.1; -0.2]);
reflected_graph = reflect(cgrph, -0.15, [0.5; 0.5], [0.25; -0.1]);
cgfx.translated_verts = translated.verts;
cgfx.translated_right_verts = translated_right.verts;
cgfx.transformed_verts = transformed.verts;
cgfx.scaled_left_verts = scaled_left.verts;
cgfx.scaled_right_verts = scaled_right.verts;
cgfx.translated = local_pack_chunkgraph(translated);
cgfx.translated_right = local_pack_chunkgraph(translated_right);
cgfx.transformed = local_pack_chunkgraph(transformed);
cgfx.scaled_left = local_pack_chunkgraph(scaled_left);
cgfx.scaled_right = local_pack_chunkgraph(scaled_right);
cgfx.rotated_verts = rotated_graph.verts;
cgfx.reflected_verts = reflected_graph.verts;

line_params = [];
line_params.ifclosed = false;
line_params.ifrefine = false;
line_params.nchmin = 2;
line_params.eps = 1.0e-3;
line_params.ta = 0.0;
line_params.tb = 1.0;
line_pref = [];
line_pref.k = 8;
[open_line, ~] = chunkerfunc(@(t) chnk.curves.linefunc(t, [2.0; -0.5], [3.0; 0.2]), line_params, line_pref);
to_closed = tochunkgraph(circ1);
to_open = tochunkgraph(open_line);
cgfx.tochunkgraph_closed = local_pack_chunkgraph(to_closed);
cgfx.tochunkgraph_open = local_pack_chunkgraph(to_open);
cgfx.tochunkgraph_closed_edge_count = numel(to_closed.echnks);
cgfx.tochunkgraph_open_edge_count = numel(to_open.echnks);

cgfx.region_points = [
    0.5, 1.5, 0.25;
    0.5, 0.5, 1.25
];
cgfx.inregion_points = chunkgraphinregion(cgrph, cgfx.region_points);
cgfx.region_x = [-0.25, 0.5, 1.25];
cgfx.region_y = [0.25, 0.75];
cgfx.inregion_grid = chunkgraphinregion(cgrph, {cgfx.region_x, cgfx.region_y});
geometry_core.chunkgraph = cgfx;

save(fullfile(outdir, 'geometry_core.mat'), 'geometry_core');

function r = local_bad_shape(t)
    r = ones(2, numel(t) + 1);
end

function [r, d] = local_bad_dim(t)
    r = ones(2, numel(t));
    d = ones(3, numel(t));
end

function [r, d, d2] = local_circle(t, ctr, radius)
    tt = t(:).';
    r = ctr + radius*[cos(tt); sin(tt)];
    d = radius*[-sin(tt); cos(tt)];
    d2 = radius*[-cos(tt); -sin(tt)];
end

function fields = local_pack_chunkgraph(cg)
    fields = [];
    fields.verts = cg.verts;
    fields.edgesendverts = cg.edgesendverts;
    fields.v2emat = full(cg.v2emat);
    fields.npt = cg.npt;
    fields.r = cg.r;
    fields.d = cg.d;
    fields.d2 = cg.d2;
    fields.n = cg.n;
    fields.wts = cg.wts;
    fields.adj = cg.adj;
    fields.min = min(cg);
    fields.max = max(cg);
    fields.edge_npt = zeros(1, numel(cg.echnks));
    for iedge = 1:numel(cg.echnks)
        fields.edge_npt(iedge) = cg.echnks(iedge).npt;
    end
end
