import numpy as np

from muFFTTO.analytical_grid_adaptation import (
    adapt_grid_to_circle,
    make_grid_nodes,
    project_points_to_circle,
    outer_boundary_mask,
    stiffness_from_distance,
    manhattan_distance_to_interface,
    cell_labels,
    interface_node_mask,
)


def test_make_grid_nodes_shape():
    """
    Function that tests the basic shape of the generated node grid.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    P, x, y = make_grid_nodes(N=8, L=2.0)
    assert P.shape == (9, 9, 2)
    assert x.shape == (9,)
    assert y.shape == (9,)


def test_project_points_to_circle_lands_on_radius():
    """
    Function that verifies projection onto the circle radius.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pts = np.array([[3.0, 4.0], [5.0, 0.0], [-8.0, 6.0]])
    out = project_points_to_circle(pts, center=(0.0, 0.0), R=10.0)
    radii = np.linalg.norm(out, axis=1)
    assert np.allclose(radii, 10.0, atol=1e-10)


def test_interface_exists_for_circle_case():
    """
    Function that checks if interface nodes exist for a circular inclusion.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    P, _, _ = make_grid_nodes(N=32, L=10.0)
    inside = cell_labels(P, center=(0.0, 0.0), R=5.0)
    interface = interface_node_mask(inside)
    assert interface.any()


def test_outer_boundary_mask_marks_edges_only():
    """
    Function that checks the correctness of the outer-boundary mask.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    N = 6
    mask = outer_boundary_mask(N)
    assert mask[0, :].all() and mask[-1, :].all()
    assert mask[:, 0].all() and mask[:, -1].all()
    assert not mask[1:-1, 1:-1].any()


def test_manhattan_distance_zero_on_interface():
    """
    Function that tests the Manhattan distance field for a single interface node.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    interface = np.zeros((5, 5), dtype=bool)
    interface[2, 2] = True
    dist = manhattan_distance_to_interface(interface)
    assert dist[2, 2] == 0.0
    assert dist[2, 3] == 1.0
    assert dist[0, 0] == 4.0


def test_stiffness_is_non_increasing_with_distance_and_has_floor():
    """
    Function that validates the stiffness decay law.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dist = np.array([0.0, 1.0, 2.0, 3.0, 50.0])
    k = stiffness_from_distance(dist, k0=1.0, b=2.0, a=1.0, kmin=0.02)
    assert np.all(k[:-1] >= k[1:])
    assert np.all(k >= 0.02)


def test_adapt_grid_moves_some_free_nodes_and_keeps_boundary_nodes():
    """
    Function that checks grid adaptation behaviour for boundary and free nodes.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    N = 24
    result = adapt_grid_to_circle(N=N, L=20.0, R=8.0, iters=80, omega=0.8)
    P0 = result["P0"]
    P = result["P"]
    interface = result["interface"]

    boundary = outer_boundary_mask(N)
    free = ~(boundary | interface)

    disp = np.linalg.norm(P - P0, axis=2)

    assert np.allclose(disp[boundary], 0.0)
    assert np.any(disp[free] > 1e-10)


def test_interface_nodes_lie_on_circle_after_adaptation():
    """
    Function that validates the circle radius for adapted interface nodes.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    result = adapt_grid_to_circle(N=24, L=20.0, R=8.0, iters=10, omega=0.8)
    P = result["P"]
    interface = result["interface"]
    pts = P[interface]
    radii = np.linalg.norm(pts, axis=1)
    assert np.allclose(radii, 8.0, atol=1e-8)
    