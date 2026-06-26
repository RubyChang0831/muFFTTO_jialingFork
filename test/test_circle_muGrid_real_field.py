import numpy as np
import pytest

from muFFTTO.circle_muGrid_real_field import (
    make_grid_nodes,
    cell_labels,
    interface_node_mask,
    project_points_to_circle,
    manhattan_distance_to_interface,
    stiffness_from_distance,
    spring_relax_weighted,
    adapt_grid_to_circle,
    make_mugrid_decomposition,
    _copy_array_into_field_p,
    make_numpy_field_bundle,
    pack_adapted_grid_to_fields,
    pack_useful_fields_to_mugrid,
)

ATOL_STRICT = 1e-12

BASELINE = {
    "N": 16,
    "L": 40.0,
    "center": (20.0, 20.0),
    "R": 10.0,
    "iters": 50,
    "omega": 0.8,
    "b": 1.0,
    "k0": 0.25,
    "kmin": 0.005,
    "interface_count": 32,
    "interface_max_abs_radius_error": 1.7763568394002505e-15,
    "interface_mean_abs_radius_error": 2.220446049250313e-16,
    "non_interface_min_abs_radius_error": 0.4009862042238499,
    "max_displacement": 3.4982306694783114,
}


def _run_baseline_case():
    result = adapt_grid_to_circle(
        N=BASELINE["N"],
        L=BASELINE["L"],
        center=BASELINE["center"],
        R=BASELINE["R"],
        iters=BASELINE["iters"],
        omega=BASELINE["omega"],
        b=BASELINE["b"],
        k0=BASELINE["k0"],
        kmin=BASELINE["kmin"],
    )

    center = np.array(BASELINE["center"])
    R = BASELINE["R"]

    P0 = result["P0"]
    P = result["P"]
    interface = result["interface"]
    non_interface = ~interface

    pts_i = P[:, interface]
    radii_i = np.sqrt((pts_i[0] - center[0]) ** 2 + (pts_i[1] - center[1]) ** 2)
    err_i = np.abs(radii_i - R)

    pts_n = P[:, non_interface]
    radii_n = np.sqrt((pts_n[0] - center[0]) ** 2 + (pts_n[1] - center[1]) ** 2)
    err_n = np.abs(radii_n - R)

    disp = np.sqrt(((P - P0) ** 2).sum(axis=0))

    return result, err_i, err_n, disp


def test_make_grid_nodes_shape_and_spacing():
    P, x, y = make_grid_nodes(N=4, L=1.0)

    assert P.shape == (2, 4, 4)
    assert x.shape == (4,)
    assert y.shape == (4,)

    expected = np.array([0.0, 0.25, 0.5, 0.75])
    assert np.allclose(x, expected, atol=ATOL_STRICT)
    assert np.allclose(y, expected, atol=ATOL_STRICT)
    assert np.allclose(P[:, 0, 0], [0.0, 0.0], atol=ATOL_STRICT)
    assert np.allclose(P[:, 1, 0], [0.25, 0.0], atol=ATOL_STRICT)
    assert np.allclose(P[:, 0, 1], [0.0, 0.25], atol=ATOL_STRICT)


def test_cell_labels_returns_binary_inside_map():
    P, _, _ = make_grid_nodes(N=4, L=4.0)
    inside = cell_labels(P, center=(2.0, 2.0), R=0.8)

    assert inside.shape == (4, 4)
    assert inside.dtype == np.int32
    assert 0 < inside.sum() < 16


def test_interface_node_mask_handles_uniform_and_nonuniform_cases():
    uniform = np.zeros((6, 6), dtype=np.int32)
    interface_uniform = interface_node_mask(uniform)
    assert interface_uniform.shape == (6, 6)
    assert interface_uniform.dtype == bool
    assert np.count_nonzero(interface_uniform) == 0

    one_cell = np.zeros((5, 5), dtype=np.int32)
    one_cell[2, 2] = 1
    interface_nonuniform = interface_node_mask(one_cell)
    assert interface_nonuniform.shape == (5, 5)
    assert interface_nonuniform.dtype == bool
    assert np.count_nonzero(interface_nonuniform) > 0


@pytest.mark.parametrize(
    "pts",
    [
        np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 3.0]]),   # (2, k)
        np.array([[2.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), # (k, 2)
    ],
)
def test_project_points_to_circle_projects_to_exact_radius(pts):
    center = (1.0, 1.0)
    R = 2.0
    proj = project_points_to_circle(pts, center=center, R=R)

    assert proj.shape == pts.shape

    if pts.shape[0] == 2:
        radii = np.sqrt((proj[0] - center[0]) ** 2 + (proj[1] - center[1]) ** 2)
    else:
        radii = np.sqrt((proj[:, 0] - center[0]) ** 2 + (proj[:, 1] - center[1]) ** 2)

    assert np.allclose(radii, R, atol=ATOL_STRICT)


def test_project_points_to_circle_raises_for_bad_shape():
    pts = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        project_points_to_circle(pts, center=(0.0, 0.0), R=1.0)


def test_manhattan_distance_to_interface_single_seed_periodic():
    interface = np.zeros((5, 5), dtype=bool)
    interface[0, 0] = True

    dist = manhattan_distance_to_interface(interface)

    assert dist.shape == (5, 5)
    assert dist[0, 0] == pytest.approx(0.0, abs=ATOL_STRICT)
    assert dist[1, 0] == pytest.approx(1.0, abs=ATOL_STRICT)
    assert dist[0, 1] == pytest.approx(1.0, abs=ATOL_STRICT)
    assert dist[4, 0] == pytest.approx(1.0, abs=ATOL_STRICT)
    assert dist[0, 4] == pytest.approx(1.0, abs=ATOL_STRICT)


def test_stiffness_from_distance_matches_formula_and_floor():
    dist = np.array([[0.0, 1.0, 9.0]])
    k = stiffness_from_distance(dist, k0=1.0, b=1.0, a=1.0, kmin=0.15)

    expected = np.array([[1.0, 0.5, 0.15]])
    assert np.allclose(k, expected, atol=ATOL_STRICT)


def test_spring_relax_weighted_keeps_fixed_nodes_unchanged():
    P, _, _ = make_grid_nodes(N=5, L=5.0)
    fixed_mask = np.zeros((5, 5), dtype=bool)
    fixed_mask[2, 2] = True

    P_mod = P.copy()
    P_mod[:, 2, 2] = np.array([2.25, 1.75])

    k_node = np.ones((5, 5), dtype=float)
    P_relaxed = spring_relax_weighted(
        P_mod, fixed_mask=fixed_mask, k_node=k_node, iters=20, omega=0.8
    )

    assert np.allclose(P_relaxed[:, 2, 2], P_mod[:, 2, 2], atol=ATOL_STRICT)


def test_spring_relax_weighted_preserves_uniform_grid_when_free():
    P, _, _ = make_grid_nodes(N=5, L=5.0)
    fixed_mask = np.zeros((5, 5), dtype=bool)
    k_node = np.ones((5, 5), dtype=float)

    P_relaxed = spring_relax_weighted(
        P, fixed_mask=fixed_mask, k_node=k_node, iters=20, omega=1.0
    )

    assert np.allclose(P_relaxed, P, atol=ATOL_STRICT)


def test_adapt_grid_to_circle_baseline_regression():
    result, err_i, err_n, disp = _run_baseline_case()

    assert set(result.keys()) == {"P0", "P", "inside", "interface", "fixed", "dist", "k_node", "params"}
    assert result["P0"].shape == (2, 16, 16)
    assert result["P"].shape == (2, 16, 16)
    assert result["inside"].shape == (16, 16)
    assert result["interface"].shape == (16, 16)
    assert result["fixed"].shape == (16, 16)
    assert result["dist"].shape == (16, 16)
    assert result["k_node"].shape == (16, 16)

    assert err_i.size == BASELINE["interface_count"]
    assert err_i.max() == pytest.approx(BASELINE["interface_max_abs_radius_error"], abs=1e-14)
    assert err_i.mean() == pytest.approx(BASELINE["interface_mean_abs_radius_error"], abs=1e-14)
    assert err_n.min() == pytest.approx(BASELINE["non_interface_min_abs_radius_error"], abs=1e-12)
    assert disp.max() == pytest.approx(BASELINE["max_displacement"], abs=1e-12)


def test_make_mugrid_decomposition_creates_fields_with_expected_shapes():
    _, decomp = make_mugrid_decomposition(nx=8, ny=10, ghosts=1)
    field = decomp.real_field("phi", components=(2,))

    assert np.asarray(field.p).shape == (2, 8, 10)
    assert np.asarray(field.pg).shape == (2, 10, 12)


def test_copy_array_into_field_p_copies_exact_values():
    _, decomp = make_mugrid_decomposition(nx=4, ny=5, ghosts=1)
    field = decomp.real_field("phi", components=(2,))
    arr = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)

    _copy_array_into_field_p(field, arr)

    assert np.allclose(np.asarray(field.p), arr, atol=ATOL_STRICT)


def test_copy_array_into_field_p_raises_on_shape_mismatch():
    _, decomp = make_mugrid_decomposition(nx=4, ny=5, ghosts=1)
    field = decomp.real_field("phi", components=(2,))
    arr = np.zeros((2, 5, 4), dtype=float)

    with pytest.raises(ValueError):
        _copy_array_into_field_p(field, arr)


def test_make_numpy_field_bundle_contains_expected_content():
    result, _, _, _ = _run_baseline_case()

    bundle_with_pos = make_numpy_field_bundle(result, store_positions=True)
    assert set(bundle_with_pos.keys()) == {
        "displacement", "inside", "interface", "fixed", "dist", "k_node", "P0", "P"
    }
    assert bundle_with_pos["displacement"].shape == (2, 16, 16)
    assert bundle_with_pos["inside"].shape == (1, 16, 16)
    assert bundle_with_pos["P0"].shape == (2, 16, 16)
    assert bundle_with_pos["P"].shape == (2, 16, 16)

    bundle_no_pos = make_numpy_field_bundle(result, store_positions=False)
    assert "P0" not in bundle_no_pos
    assert "P" not in bundle_no_pos
    assert "displacement" in bundle_no_pos


def test_pack_adapted_grid_to_fields_writes_consistent_values():
    result, _, _, _ = _run_baseline_case()

    bundle = pack_adapted_grid_to_fields(
        result,
        ghosts=1,
        store_positions=True,
        verbose=False,
    )

    expected_names = ["displacement", "inside", "interface", "fixed", "dist", "k_node", "P0", "P"]
    for name in expected_names:
        assert bundle["report"][name]["status"] == "ok"
        assert np.allclose(np.asarray(bundle["fields"][name].p), bundle["numpy"][name], atol=ATOL_STRICT)

    displacement_expected = result["P"] - result["P0"]
    assert np.allclose(bundle["numpy"]["displacement"], displacement_expected, atol=ATOL_STRICT)

    disp_p = np.asarray(bundle["fields"]["displacement"].p)
    disp_pg = np.asarray(bundle["fields"]["displacement"].pg)
    assert disp_p.shape == (2, 16, 16)
    assert disp_pg.shape == (2, 18, 18)


def test_pack_useful_fields_to_mugrid_returns_selected_fields_only():
    result, _, _, _ = _run_baseline_case()
    bundle = pack_useful_fields_to_mugrid(result, ghosts=1, verbose=False)

    expected_keys = {"displacement", "inside", "interface", "dist", "k_node"}
    assert set(bundle["fields"].keys()) == expected_keys
    assert set(bundle["numpy"].keys()) == expected_keys
    assert set(bundle["report"].keys()) == expected_keys