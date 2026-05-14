import numpy as np
import pytest

from _fixture_generation import chunker_from_fields, load_generated_mat_fixture
from chunkie import ChunkGraph, kernel
from chunkie.quadrature import rcip

pytestmark = [pytest.mark.slow, pytest.mark.requires_matlab]


def load_fixture(name: str):
    return load_generated_mat_fixture(name, squeeze_me=True, struct_as_record=False)


def as_1d(value, dtype=None) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def scalar_int(value) -> int:
    return int(as_1d(value)[0])


def matlab_indices0(value) -> np.ndarray:
    return as_1d(value, int) - 1


def test_rcip_setup_helpers_match_matlab_fixture():
    fixture = load_fixture("rcip.mat")["rcip_fixture"]

    ip_case = fixture.IPinit
    ip, ipw = rcip.IPinit(nodes=ip_case.T, weights=ip_case.W)
    np.testing.assert_allclose(ip, ip_case.IP, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(ipw, ip_case.IPW, rtol=1e-13, atol=1e-14)

    pbc_case = fixture.Pbcinit
    pbc = rcip.Pbcinit(
        interpolation=pbc_case.IP,
        edge_count=scalar_int(pbc_case.nedge),
        dimension=scalar_int(pbc_case.ndim),
    )
    np.testing.assert_allclose(pbc, pbc_case.Pbc, rtol=1e-13, atol=1e-14)

    setup_case = fixture.setup
    actual = rcip.setup(
        quadrature_order=scalar_int(setup_case.ngl),
        dimension=scalar_int(setup_case.ndim),
        edge_count=scalar_int(setup_case.nedge),
        starts_at_corner=as_1d(setup_case.isstart, bool),
    )
    names = ("Pbc", "PWbc", "starL", "circL", "starS", "circS", "ilist", "starL1", "circL1")
    actual_by_name = dict(zip(names, actual, strict=True))
    np.testing.assert_allclose(actual_by_name["Pbc"], setup_case.Pbc, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(actual_by_name["PWbc"], setup_case.PWbc, rtol=1e-13, atol=1e-14)
    for name in ("starL", "circL", "starS", "circS", "starL1", "circL1"):
        np.testing.assert_array_equal(
            actual_by_name[name], matlab_indices0(getattr(setup_case, name))
        )
    np.testing.assert_array_equal(
        actual_by_name["ilist"], np.asarray(setup_case.ilist, dtype=int) - 1
    )


def test_rcip_schurbana_matches_matlab_fixture():
    fixture = load_fixture("rcip.mat")["rcip_fixture"]
    case = fixture.SchurBana

    args = (
        case.Pbc,
        case.PWbc,
        case.K,
        case.A_input,
        matlab_indices0(case.starL),
        matlab_indices0(case.circL),
        matlab_indices0(case.starS),
        matlab_indices0(case.circS),
    )
    actual = rcip.SchurBana(*args)

    np.testing.assert_allclose(actual, case.A_output, rtol=1e-12, atol=1e-13)


def test_corner_refine_matches_matlab_corner_topology_fixture():
    fixture = load_fixture("rcip.mat")["rcip_fixture"]
    case = fixture.corner_refine

    cg = ChunkGraph(
        np.asarray(case.verts, dtype=float),
        np.asarray(case.edgesendverts0, dtype=int),
        pref={"k": scalar_int(case.k)},
        cparams={"_chunkie_normalized_geometry_options": True, "nchmin": scalar_int(case.nchmin)},
    )
    vertex = scalar_int(case.vertex0)
    depth = scalar_int(case.depth)
    edges, signs = cg.vstruc[vertex]

    np.testing.assert_array_equal([edge.nch for edge in cg.echnks], as_1d(case.original_nch, int))
    np.testing.assert_array_equal(edges, as_1d(case.vstruc_edges0, int))
    np.testing.assert_array_equal(signs, as_1d(case.vstruc_signs, int))

    refined = rcip.corner_refine(cg, vertices=[vertex], depth=depth)
    np.testing.assert_array_equal(
        [edge.nch for edge in refined.echnks], as_1d(case.expected_nch, int)
    )


def test_chunkgraph_rcip_driver_matches_matlab_fixture():
    fixture = load_fixture("rcip.mat")["rcip_fixture"]
    case = fixture.chunkgraph_rcip
    edge1 = chunker_from_fields(fixture.edge1)
    edge2 = chunker_from_fields(fixture.edge2)
    cg = ChunkGraph(
        np.asarray(case.verts, dtype=float),
        np.asarray(case.edgesendverts0, dtype=int),
        [edge1, edge2],
    )

    result = rcip.chunkgraph_rcip(
        graph=cg,
        kernel=kernel("lap", "d"),
        dimension=scalar_int(case.ndim),
        vertices=as_1d(case.vertices0, int),
        options={
            "_chunkie_normalized_operator_options": True,
            "nsub": scalar_int(case.nsub),
            "rcip_savedepth": scalar_int(case.rcip_savedepth),
        },
    )

    assert isinstance(result, rcip.RCIPChunkGraphResult)
    assert set(result.__dataclass_fields__) == {"vertices", "edge_indices", "R", "saved", "kernels"}
    np.testing.assert_array_equal(result.vertices, as_1d(case.vertices0, int))
    assert len(result.edge_indices) == 1
    np.testing.assert_array_equal(result.edge_indices[0], as_1d(case.edge_indices0, int))
    np.testing.assert_allclose(result.R[0], case.R, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.saved[0].R[-1], case.saved_R_final, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.saved[0].MAT[-1], case.saved_MAT_last, rtol=1e-12, atol=1e-13)
