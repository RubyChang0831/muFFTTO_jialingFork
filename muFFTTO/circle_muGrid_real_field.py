"""
Storage into muGrid
real_field containers through the primary accessor `field.p`.

Reference:
Zecevic, M., Lebensohn, R. A., & Capolungo, L. (2026).
Achieving geometric accuracy in FFT-based micromechanical models
using conformal grid. Mechanics of Materials, 212, 105512.
https://doi.org/10.1016/j.mechmat.2025.105512
"""

from __future__ import annotations
from collections import deque
import numpy as np
import muGrid

def make_grid_nodes(N: int = 64, L: float = 25.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function that constructs a regular periodic-style Cartesian node grid.

    The grid stores N x N nodal points on the square domain [0, L) x [0, L).
    The right and top boundary nodes are not stored explicitly, which is consistent
    with a periodic grid representation. Node coordinates are stored in an array
    of shape [xy, nx, ny].

    Parameters
    ----------
    N : int
        Number of stored grid points per spatial direction.
    L : float
        Side length of the square computational domain measured in the same units
        as x and y.

    Returns
    -------
    P : numpy.ndarray
        Array of nodal coordinates with shape [xy, nx, ny].
        - xy = 2 stores the x- and y-coordinates.
        - nx = N is the number of stored nodes in x-direction.
        - ny = N is the number of stored nodes in y-direction.
    x : numpy.ndarray
        One-dimensional array of x-coordinates with shape [nx].
    y : numpy.ndarray
        One-dimensional array of y-coordinates with shape [ny].
    """
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.stack([X, Y], axis=0)
    return P, x, y


def cell_labels(
    P: np.ndarray,
    center: tuple[float, float] = (0.0, 0.0),
    R: float = 20.0,
) -> np.ndarray:
    """
    Function that labels cells as inside or outside a circular inclusion.

    Cell centers are computed from the four corner nodes of each periodic cell.
    A cell is marked as inside if its center lies inside or on the target circle.

    Parameters
    ----------
    P : numpy.ndarray
        Array of nodal coordinates with shape [xy, nx, ny].
    center : tuple[float, float]
        Coordinates of the circle center given as (x_center, y_center).
    R : float
        Radius of the circular inclusion.

    Returns
    -------
    inside : numpy.ndarray
        Integer array with shape [nx, ny].
        A value of 1 indicates that the periodic cell center lies inside the circle,
        and a value of 0 indicates that it lies outside.
    """
    cx, cy = center

    P_ip1 = np.roll(P, shift=-1, axis=1)
    P_jp1 = np.roll(P, shift=-1, axis=2)
    P_ip1_jp1 = np.roll(P_ip1, shift=-1, axis=2)

    P_ip1 = P_ip1.copy()
    P_jp1 = P_jp1.copy()
    P_ip1_jp1 = P_ip1_jp1.copy()

    dx_grid = P[0, 1, 0] - P[0, 0, 0]
    dy_grid = P[1, 0, 1] - P[1, 0, 0]

    P_ip1[0, -1, :] += dx_grid * P.shape[1]
    P_ip1_jp1[0, -1, :] += dx_grid * P.shape[1]
    P_jp1[1, :, -1] += dy_grid * P.shape[2]
    P_ip1_jp1[1, :, -1] += dy_grid * P.shape[2]

    Pc = 0.25 * (P + P_ip1 + P_jp1 + P_ip1_jp1)
    dx = Pc[0] - cx
    dy = Pc[1] - cy
    inside_mask = (dx * dx + dy * dy) <= R * R
    return inside_mask.astype(np.int32)


def interface_node_mask(cell_inside: np.ndarray) -> np.ndarray:
    """
    Function that detects interface nodes on a periodic stored grid.

    A stored node is marked as an interface node if the surrounding periodic cells
    do not all have the same inside/outside label. In other words, the node lies
    on the discrete boundary between two material regions.

    Parameters
    ----------
    cell_inside : numpy.ndarray
        Integer array of cell labels with shape [nx, ny].
        Cells with value 1 are inside the inclusion and cells with value 0 are outside.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array with shape [nx, ny].
        True indicates that the node is an interface node, and False otherwise.
    """
    N = cell_inside.shape[0]
    mask = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(N):
            vals = np.array([
                cell_inside[(i - 1) % N, (j - 1) % N],
                cell_inside[i % N, (j - 1) % N],
                cell_inside[(i - 1) % N, j % N],
                cell_inside[i % N, j % N],
            ], dtype=int)
            if vals.min() != vals.max():
                mask[i, j] = True

    return mask


def project_points_to_circle(
    Ppts: np.ndarray,
    center: tuple[float, float] = (0.0, 0.0),
    R: float = 20.0,
) -> np.ndarray:
    """
    Function that projects points onto a target circle.

    Each input point is moved along the radial direction so that its distance
    from the circle center becomes exactly equal to the prescribed radius.
    The function accepts point arrays in either [xy, k] or [k, xy] format.

    Parameters
    ----------
    Ppts : numpy.ndarray
        Array of point coordinates with shape [xy, k] or [k, xy].
    center : tuple[float, float]
        Coordinates of the circle center given as (x_center, y_center).
    R : float
        Radius of the target circle.

    Returns
    -------
    proj : numpy.ndarray
        Array of projected point coordinates with the same shape convention
        as the input array `Ppts`.

    Raises
    ------
    ValueError
        If `Ppts` is not a two-dimensional array with shape [xy, k] or [k, xy].
    """
    c = np.asarray(center, dtype=float)

    transposed = False
    if Ppts.ndim != 2:
        raise ValueError("Ppts must be a 2D array")
    if Ppts.shape[0] == 2:
        pts = Ppts.T
        transposed = True
    elif Ppts.shape[1] == 2:
        pts = Ppts
    else:
        raise ValueError("Ppts must have shape [xy, k] or [k, xy]")

    V = pts - c[None, :]
    r = np.linalg.norm(V, axis=1, keepdims=True)
    r = np.maximum(r, 1e-12)
    proj = c[None, :] + (R / r) * V
    return proj.T if transposed else proj


def manhattan_distance_to_interface(interface_mask: np.ndarray) -> np.ndarray:
    """
    Function that computes the periodic Manhattan distance to the nearest interface node.

    The distance is measured in the discrete grid sense using nearest-neighbor
    connectivity in the x- and y-directions. Periodic wrapping is applied at the
    domain boundaries.

    Parameters
    ----------
    interface_mask : numpy.ndarray
        Boolean array with shape [nx, ny].
        True marks interface nodes and False marks non-interface nodes.

    Returns
    -------
    dist : numpy.ndarray
        Floating-point array with shape [nx, ny].
        Each entry contains the periodic Manhattan distance from that node to
        the nearest interface node.
    """
    N = interface_mask.shape[0]
    dist = np.full((N, N), np.inf, dtype=float)
    queue = deque()

    for i in range(N):
        for j in range(N):
            if interface_mask[i, j]:
                dist[i, j] = 0.0
                queue.append((i, j))

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        i, j = queue.popleft()
        for di, dj in neighbors:
            ii = (i + di) % N
            jj = (j + dj) % N
            if dist[ii, jj] > dist[i, j] + 1.0:
                dist[ii, jj] = dist[i, j] + 1.0
                queue.append((ii, jj))

    return dist


def stiffness_from_distance(
    dist: np.ndarray,
    k0: float = 1.0,
    b: float = 1.0,
    a: float = 1.0,
    kmin: float = 1e-3,
) -> np.ndarray:
    """
    Function that computes a nodal stiffness field from the distance to the interface.

    The stiffness is defined by a distance-dependent power law and is bounded
    from below by a prescribed minimum stiffness value.

    Parameters
    ----------
    dist : numpy.ndarray
        Array of nodal distances to the nearest interface with shape [nx, ny].
    k0 : float
        Reference stiffness factor.
    b : float
        Exponent controlling how fast the stiffness decays with distance.
    a : float
        Positive shift added to the distance to avoid division by zero and to
        control the stiffness near the interface.
    kmin : float
        Minimum admissible stiffness value.

    Returns
    -------
    k : numpy.ndarray
        Floating-point stiffness array with shape [nx, ny].
    """
    k = k0 / np.power(dist + a, b)
    return np.maximum(k, kmin)


def spring_relax_weighted(
    P: np.ndarray,
    fixed_mask: np.ndarray,
    k_node: np.ndarray,
    iters: int = 600,
    omega: float = 1.0,
) -> np.ndarray:
    """
    Function that performs weighted spring relaxation on a periodic stored grid.

    Each non-fixed node is iteratively moved toward a weighted average of its
    four nearest neighbors. The nodal stiffness field controls the local strength
    of the spring interaction, and periodic wrapping is applied at the domain boundaries.

    Parameters
    ----------
    P : numpy.ndarray
        Array of nodal coordinates with shape [xy, nx, ny].
    fixed_mask : numpy.ndarray
        Boolean array with shape [nx, ny].
        True marks fixed nodes that remain unchanged during relaxation.
    k_node : numpy.ndarray
        Floating-point nodal stiffness array with shape [nx, ny].
    iters : int
        Number of relaxation iterations.
    omega : float
        Relaxation parameter. Values between 0 and 1 correspond to under-relaxation,
        and omega = 1 applies the full update at each iteration.

    Returns
    -------
    P_new : numpy.ndarray
        Relaxed nodal coordinates with shape [xy, nx, ny].
    """
    P_new = P.copy()
    N = P.shape[1]

    dx_grid = P[0, 1, 0] - P[0, 0, 0]
    dy_grid = P[1, 0, 1] - P[1, 0, 0]

    for _ in range(iters):
        for i in range(N):
            for j in range(N):
                if fixed_mask[i, j]:
                    continue

                neighbors = [
                    ((i - 1) % N, j),
                    ((i + 1) % N, j),
                    (i, (j - 1) % N),
                    (i, (j + 1) % N),
                ]
                kij = k_node[i, j]
                w = np.empty(4, dtype=float)
                pts = np.empty((4, 2), dtype=float)

                xij = P_new[0, i, j]
                yij = P_new[1, i, j]

                for idx, (ii, jj) in enumerate(neighbors):
                    xnb = P_new[0, ii, jj]
                    ynb = P_new[1, ii, jj]

                    if ii == 0 and i == N - 1:
                        xnb += dx_grid * N
                    elif ii == N - 1 and i == 0:
                        xnb -= dx_grid * N

                    if jj == 0 and j == N - 1:
                        ynb += dy_grid * N
                    elif jj == N - 1 and j == 0:
                        ynb -= dy_grid * N

                    w[idx] = 0.5 * (kij + k_node[ii, jj])
                    pts[idx] = [xnb, ynb]

                target = (w[:, None] * pts).sum(axis=0) / (w.sum() + 1e-15)
                P_new[:, i, j] = (1.0 - omega) * np.array([xij, yij]) + omega * target

    return P_new


def adapt_grid_to_circle(
    N: int = 64,
    L: float = 40.0,
    center: tuple[float, float] = (20.0, 20.0),
    R: float = 10.0,
    iters: int = 600,
    omega: float = 0.8,
    b: float = 1.0,
    k0: float = 0.25,
    kmin: float = 0.005,
) -> dict:
    """
    Function that adapts a periodic Cartesian grid to a circular inclusion.

    The procedure consists of four main steps:
    1. Construct the initial periodic grid.
    2. Label cells and detect interface nodes.
    3. Project interface nodes onto the target circle.
    4. Relax the remaining nodes with a weighted spring model.

    Parameters
    ----------
    N : int
        Number of stored grid points per spatial direction.
    L : float
        Side length of the square computational domain.
    center : tuple[float, float]
        Coordinates of the circle center given as (x_center, y_center).
    R : float
        Radius of the circular inclusion.
    iters : int
        Number of spring-relaxation iterations.
    omega : float
        Relaxation parameter for the spring-relaxation update.
    b : float
        Exponent of the stiffness-distance law.
    k0 : float
        Reference stiffness factor.
    kmin : float
        Minimum admissible stiffness value.

    Returns
    -------
    result : dict
        Dictionary containing the adapted grid and related fields:
        - "P0": initial nodal coordinates, shape [xy, nx, ny]
        - "P": adapted nodal coordinates, shape [xy, nx, ny]
        - "inside": cell labels, shape [nx, ny]
        - "interface": interface-node mask, shape [nx, ny]
        - "fixed": fixed-node mask, shape [nx, ny]
        - "dist": distance-to-interface field, shape [nx, ny]
        - "k_node": nodal stiffness field, shape [nx, ny]
        - "params": dictionary of input parameters
    """
    P0, _, _ = make_grid_nodes(N=N, L=L)
    inside = cell_labels(P0, center=center, R=R)
    interface = interface_node_mask(inside)
    fixed = interface.copy()

    P1 = P0.copy()
    idx = np.argwhere(interface)
    if idx.size:
        pts = P1[:, idx[:, 0], idx[:, 1]]
        P1[:, idx[:, 0], idx[:, 1]] = project_points_to_circle(
            pts,
            center=center,
            R=R,
        )

    dist = manhattan_distance_to_interface(interface)
    k_node = stiffness_from_distance(dist, k0=k0, b=b, a=1.0, kmin=kmin)
    P2 = spring_relax_weighted(P1, fixed_mask=fixed, k_node=k_node, iters=iters, omega=omega)

    return {
        "P0": P0,
        "P": P2,
        "inside": inside,
        "interface": interface,
        "fixed": fixed,
        "dist": dist,
        "k_node": k_node,
        "params": {
            "N": N,
            "L": L,
            "center": center,
            "R": R,
            "iters": iters,
            "omega": omega,
            "b": b,
            "k0": k0,
            "kmin": kmin,
        },
    }


def make_mugrid_decomposition(
    nx: int,
    ny: int,
    ghosts: int = 1,
):
    """
    Function that creates a simple single-process muGrid Cartesian decomposition.

    The decomposition is defined for a structured grid with a prescribed number
    of domain points and a prescribed number of ghost points on each side.

    Parameters
    ----------
    nx : int
        Number of domain grid points in x-direction.
    ny : int
        Number of domain grid points in y-direction.
    ghosts : int
        Number of ghost points added on the left and right sides of each direction.

    Returns
    -------
    comm : muGrid.Communicator
        muGrid communicator object.
    decomp : muGrid.CartesianDecomposition
        Cartesian decomposition associated with the communicator and grid layout.
    """
    comm = muGrid.Communicator()
    decomp = muGrid.CartesianDecomposition(
        communicator=comm,
        nb_domain_grid_pts=(nx, ny),
        nb_subdivisions=(1, 1),
        nb_ghosts_left=(ghosts, ghosts),
        nb_ghosts_right=(ghosts, ghosts),
    )
    return comm, decomp


def _copy_array_into_field_p(field, arr: np.ndarray) -> None:
    """
    Function that copies a NumPy array into a muGrid field through the primary accessor `p`.

    This function assumes that the muGrid field uses the shape convention
    [components, nx, ny] for vector and scalar fields stored through `field.p`.

    Parameters
    ----------
    field : muGrid.Field
        muGrid field object whose primary accessor `p` will be written.
    arr : numpy.ndarray
        Array to be copied into the field.
        Expected shapes are:
        - [ncomp, nx, ny] for vector-valued fields
        - [1, nx, ny] for scalar-valued fields stored with one component

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the shape of `arr` does not match the shape of `field.p`.
    """
    target = np.asarray(field.p)
    if target.shape != arr.shape:
        raise ValueError(
            f"Shape mismatch for field.p: target shape = {target.shape}, arr shape = {arr.shape}"
        )
    target[...] = arr


def make_numpy_field_bundle(result: dict, store_positions: bool = True) -> dict:
    """
    Function that builds a dictionary of useful NumPy fields from the grid-adaptation result.

    The returned bundle contains deformation-related scalar and vector fields
    derived from the output of `adapt_grid_to_circle`. The displacement field is
    computed as the difference between the adapted and initial nodal coordinates.

    Parameters
    ----------
    result : dict
        Dictionary returned by `adapt_grid_to_circle`.
    store_positions : bool
        If True, the initial and adapted nodal coordinates `P0` and `P` are also stored
        in the returned bundle. If False, only derived fields are stored.

    Returns
    -------
    bundle : dict
        Dictionary containing NumPy arrays:
        - "displacement": shape [2, nx, ny]
        - "inside": shape [1, nx, ny]
        - "interface": shape [1, nx, ny]
        - "fixed": shape [1, nx, ny]
        - "dist": shape [1, nx, ny]
        - "k_node": shape [1, nx, ny]
        and optionally
        - "P0": shape [2, nx, ny]
        - "P": shape [2, nx, ny]
    """
    P0 = result["P0"]
    P = result["P"]
    inside = result["inside"]
    interface = result["interface"]
    fixed = result["fixed"]
    dist = result["dist"]
    k_node = result["k_node"]

    displacement = P - P0

    bundle = {
        "displacement": displacement.astype(np.float64),
        "inside": inside[None, ...].astype(np.float64),
        "interface": interface[None, ...].astype(np.float64),
        "fixed": fixed[None, ...].astype(np.float64),
        "dist": dist[None, ...].astype(np.float64),
        "k_node": k_node[None, ...].astype(np.float64),
    }

    if store_positions:
        bundle["P0"] = P0.astype(np.float64)
        bundle["P"] = P.astype(np.float64)

    return bundle


def pack_adapted_grid_to_fields(
    result: dict,
    ghosts: int = 1,
    store_positions: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Function that stores adapted-grid data into muGrid real_field containers.

    The function first builds a NumPy field bundle from the adaptation result,
    then creates one muGrid real_field per stored quantity, and finally copies
    each NumPy array into the corresponding field through `field.p`.

    Parameters
    ----------
    result : dict
        Dictionary returned by `adapt_grid_to_circle`.
    ghosts : int
        Number of ghost points used in the muGrid decomposition.
    store_positions : bool
        If True, the initial and adapted nodal coordinates `P0` and `P` are also stored.
    verbose : bool
        If True, a short write-status message is printed for each stored field.

    Returns
    -------
    bundle : dict
        Dictionary containing:
        - "comm": muGrid communicator
        - "decomp": muGrid Cartesian decomposition
        - "fields": dictionary of muGrid real_field objects
        - "numpy": dictionary of NumPy arrays used for storage
        - "report": dictionary summarizing shapes and storage status
    """
    nx, ny = result["P"].shape[1], result["P"].shape[2]
    comm, decomp = make_mugrid_decomposition(nx=nx, ny=ny, ghosts=ghosts)

    numpy_fields = make_numpy_field_bundle(result, store_positions=store_positions)
    mugrid_fields = {}
    report = {}

    for name, arr in numpy_fields.items():
        ncomp = arr.shape[0] if arr.ndim == 3 else 1
        field = decomp.real_field(name, components=(ncomp,))
        mugrid_fields[name] = field

        _copy_array_into_field_p(field, arr)

        report[name] = {
            "status": "ok",
            "shape": arr.shape,
            "p_shape": np.asarray(field.p).shape,
            "pg_shape": np.asarray(field.pg).shape,
        }

        if verbose:
            print(
                f"[ok] field '{name}' written | "
                f"shape={arr.shape} | p_shape={np.asarray(field.p).shape}"
            )

    return {
        "comm": comm,
        "decomp": decomp,
        "fields": mugrid_fields,
        "numpy": numpy_fields,
        "report": report,
    }


def print_field_summary(field_bundle: dict) -> None:
    """
    Function that prints a summary of the muGrid fields stored in a field bundle.

    The printed summary includes the storage status and the shapes of the
    non-ghost (`p`) and ghosted (`pg`) field views for each stored quantity.

    Parameters
    ----------
    field_bundle : dict
        Dictionary returned by `pack_adapted_grid_to_fields`.

    Returns
    -------
    None
    """
    print("\nmuGrid field summary")
    print("-" * 72)

    for name, report in field_bundle["report"].items():
        print(f"{name}:")
        print(f"  status  : {report['status']}")
        print(f"  shape   : {report['shape']}")
        print(f"  p_shape : {report['p_shape']}")
        print(f"  pg_shape: {report['pg_shape']}")
        print()


def pack_useful_fields_to_mugrid(result: dict, ghosts: int = 1, verbose: bool = False) -> dict:
    """
    Function that stores only the most useful derived fields into muGrid containers.

    This is a reduced version of `pack_adapted_grid_to_fields` intended for
    downstream workflows that only require a subset of the available fields.

    Parameters
    ----------
    result : dict
        Dictionary returned by `adapt_grid_to_circle`.
    ghosts : int
        Number of ghost points used in the muGrid decomposition.
    verbose : bool
        If True, print storage information during field creation.

    Returns
    -------
    bundle : dict
        Dictionary containing only the selected fields:
        - "displacement"
        - "inside"
        - "interface"
        - "dist"
        - "k_node"
        together with the corresponding communicator, decomposition, NumPy arrays,
        and storage report.
    """
    bundle = pack_adapted_grid_to_fields(
        result,
        ghosts=ghosts,
        store_positions=False,
        verbose=verbose,
    )

    useful_names = ["displacement", "inside", "interface", "dist", "k_node"]

    return {
        "comm": bundle["comm"],
        "decomp": bundle["decomp"],
        "fields": {name: bundle["fields"][name] for name in useful_names},
        "numpy": {name: bundle["numpy"][name] for name in useful_names},
        "report": {name: bundle["report"][name] for name in useful_names},
    }


'''
//successful version

"""
The code is genreated by Perplexity AI.


Reference:
Zecevic, M., Lebensohn, R. A., & Capolungo, L. (2026).
Achieving geometric accuracy in FFT-based micromechanical models
using conformal grid. Mechanics of Materials, 212, 105512.
[https://doi.org/10.1016/j.mechmat.2025.105512](https://doi.org/10.1016/j.mechmat.2025.105512)


"""


import numpy as np
from collections import deque
import muGrid




def make_grid_nodes(N: int = 64, L: float = 25.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a periodic-style Cartesian grid on [0, L) x [0, L).


    Notes
    -----
    - The domain is interpreted as N cells per direction.
    - Only N x N grid points are stored.
    - The right and top boundary points are not stored explicitly.
    - Coordinates are stored with shape [xy, nx, ny].


    Parameters
    ----------
    N : int
        Number of cells per spatial direction.
    L : float
        Domain size.


    Returns
    -------
    P : numpy.ndarray
        Grid-point coordinates with shape [xy, nx, ny].
    x : numpy.ndarray
        x-coordinates of stored points, shape [nx].
    y : numpy.ndarray
        y-coordinates of stored points, shape [ny].
    """
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.stack([X, Y], axis=0)
    return P, x, y




def cell_labels(
    P: np.ndarray,
    center: tuple[float, float] = (0.0, 0.0),
    R: float = 20.0,
) -> np.ndarray:
    """
    Label periodic cells as inside or outside a circular inclusion.


    Cell centers are built from the stored node at (i, j) and its periodic
    neighbors (i+1, j), (i, j+1), (i+1, j+1).


    Parameters
    ----------
    P : numpy.ndarray
        Grid-point coordinates with shape [xy, nx, ny].
    center : tuple of float
        Circle center.
    R : float
        Circle radius.


    Returns
    -------
    inside : numpy.ndarray
        Integer array with shape [N, N].
    """
    cx, cy = center


    P_ip1 = np.roll(P, shift=-1, axis=1)
    P_jp1 = np.roll(P, shift=-1, axis=2)
    P_ip1_jp1 = np.roll(P_ip1, shift=-1, axis=2)


    P_ip1 = P_ip1.copy()
    P_jp1 = P_jp1.copy()
    P_ip1_jp1 = P_ip1_jp1.copy()


    P_ip1[0, -1, :] += R * 0.0
    P_jp1[1, :, -1] += R * 0.0


    P_ip1[0, -1, :] += (P[0, 1, 0] - P[0, 0, 0]) * P.shape[1]
    P_ip1_jp1[0, -1, :] += (P[0, 1, 0] - P[0, 0, 0]) * P.shape[1]
    P_jp1[1, :, -1] += (P[1, 0, 1] - P[1, 0, 0]) * P.shape[2]
    P_ip1_jp1[1, :, -1] += (P[1, 0, 1] - P[1, 0, 0]) * P.shape[2]


    Pc = 0.25 * (P + P_ip1 + P_jp1 + P_ip1_jp1)
    dx = Pc[0] - cx
    dy = Pc[1] - cy
    inside_mask = (dx * dx + dy * dy) <= R * R
    return inside_mask.astype(np.int32)




def interface_node_mask(cell_inside: np.ndarray) -> np.ndarray:
    """
    Detect interface nodes on a periodic N x N stored grid.


    A stored point is marked if the surrounding periodic cells do not all have
    the same label.
    """
    N = cell_inside.shape[0]
    mask = np.zeros((N, N), dtype=bool)


    for i in range(N):
        for j in range(N):
            vals = np.array([
                cell_inside[(i - 1) % N, (j - 1) % N],
                cell_inside[i % N, (j - 1) % N],
                cell_inside[(i - 1) % N, j % N],
                cell_inside[i % N, j % N],
            ], dtype=int)
            if vals.min() != vals.max():
                mask[i, j] = True


    return mask




def project_points_to_circle(
    Ppts: np.ndarray,
    center: tuple[float, float] = (0.0, 0.0),
    R: float = 20.0,
) -> np.ndarray:
    """
    Project points onto the target circle.


    Accepts arrays with shape [xy, k] or [k, xy].
    """
    c = np.asarray(center, dtype=float)


    transposed = False
    if Ppts.ndim != 2:
        raise ValueError("Ppts must be a 2D array")
    if Ppts.shape[0] == 2:
        pts = Ppts.T
        transposed = True
    elif Ppts.shape[1] == 2:
        pts = Ppts
    else:
        raise ValueError("Ppts must have shape [xy, k] or [k, xy]")


    V = pts - c[None, :]
    r = np.linalg.norm(V, axis=1, keepdims=True)
    r = np.maximum(r, 1e-12)
    proj = c[None, :] + (R / r) * V
    return proj.T if transposed else proj




def manhattan_distance_to_interface(interface_mask: np.ndarray) -> np.ndarray:
    """
    Compute periodic Manhattan distance to the nearest interface point.
    """
    N = interface_mask.shape[0]
    dist = np.full((N, N), np.inf, dtype=float)
    queue = deque()


    for i in range(N):
        for j in range(N):
            if interface_mask[i, j]:
                dist[i, j] = 0.0
                queue.append((i, j))


    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        i, j = queue.popleft()
        for di, dj in neighbors:
            ii = (i + di) % N
            jj = (j + dj) % N
            if dist[ii, jj] > dist[i, j] + 1.0:
                dist[ii, jj] = dist[i, j] + 1.0
                queue.append((ii, jj))


    return dist




def stiffness_from_distance(
    dist: np.ndarray,
    k0: float = 1.0,
    b: float = 1.0,
    a: float = 1.0,
    kmin: float = 1e-3,
) -> np.ndarray:
    """
    Compute nodal stiffness from distance to the interface.
    """
    k = k0 / np.power(dist + a, b)
    return np.maximum(k, kmin)




def spring_relax_weighted(
    P: np.ndarray,
    fixed_mask: np.ndarray,
    k_node: np.ndarray,
    iters: int = 600,
    omega: float = 1.0,
) -> np.ndarray:
    """
    Perform weighted spring relaxation on a periodic stored grid.


    Parameters
    ----------
    P : numpy.ndarray
        Grid-point coordinates with shape [xy, nx, ny].
    fixed_mask : numpy.ndarray
        Boolean mask with shape [nx, ny].
    k_node : numpy.ndarray
        Stiffness field with shape [nx, ny].
    iters : int
        Number of relaxation sweeps.
    omega : float
        Relaxation parameter.


    Returns
    -------
    P_relaxed : numpy.ndarray
        Relaxed coordinates with shape [xy, nx, ny].
    """
    P_new = P.copy()
    N = P.shape[1]


    for _ in range(iters):
        for i in range(N):
            for j in range(N):
                if fixed_mask[i, j]:
                    continue


                neighbors = [
                    ((i - 1) % N, j),
                    ((i + 1) % N, j),
                    (i, (j - 1) % N),
                    (i, (j + 1) % N),
                ]
                kij = k_node[i, j]
                w = np.empty(4, dtype=float)
                pts = np.empty((4, 2), dtype=float)


                xij = P_new[0, i, j]
                yij = P_new[1, i, j]


                for idx, (ii, jj) in enumerate(neighbors):
                    xnb = P_new[0, ii, jj]
                    ynb = P_new[1, ii, jj]


                    if ii == 0 and i == N - 1:
                        xnb += P[0, 1, 0] - P[0, 0, 0]
                        xnb += (N - 1) * (P[0, 1, 0] - P[0, 0, 0])
                    elif ii == N - 1 and i == 0:
                        xnb -= P[0, 1, 0] - P[0, 0, 0]
                        xnb -= (N - 1) * (P[0, 1, 0] - P[0, 0, 0])


                    if jj == 0 and j == N - 1:
                        ynb += P[1, 0, 1] - P[1, 0, 0]
                        ynb += (N - 1) * (P[1, 0, 1] - P[1, 0, 0])
                    elif jj == N - 1 and j == 0:
                        ynb -= P[1, 0, 1] - P[1, 0, 0]
                        ynb -= (N - 1) * (P[1, 0, 1] - P[1, 0, 0])


                    w[idx] = 0.5 * (kij + k_node[ii, jj])
                    pts[idx] = [xnb, ynb]


                target = (w[:, None] * pts).sum(axis=0) / (w.sum() + 1e-15)
                P_new[:, i, j] = (1.0 - omega) * np.array([xij, yij]) + omega * target


    return P_new




def adapt_grid_to_circle(
    N: int = 64,
    L: float = 40.0,
    center: tuple[float, float] = (20.0, 20.0),
    R: float = 10.0,
    iters: int = 600,
    omega: float = 0.8,
    b: float = 1.0,
    k0: float = 0.25,
    kmin: float = 0.005,
) -> dict:
    """
    Adapt a periodic-style grid to a circular inclusion.


    Notes
    -----
    - There are N cells and only N x N stored points.
    - Right and top boundaries are not stored explicitly.
    - All coordinate arrays use shape [xy, nx, ny].
    - Interface detection and distance are treated periodically.
    """
    P0, _, _ = make_grid_nodes(N=N, L=L)
    inside = cell_labels(P0, center=center, R=R)
    interface = interface_node_mask(inside)
    fixed = interface.copy()


    P1 = P0.copy()
    idx = np.argwhere(interface)
    if idx.size:
        pts = P1[:, idx[:, 0], idx[:, 1]]
        P1[:, idx[:, 0], idx[:, 1]] = project_points_to_circle(
            pts,
            center=center,
            R=R,
        )


    dist = manhattan_distance_to_interface(interface)
    k_node = stiffness_from_distance(dist, k0=k0, b=b, a=1.0, kmin=kmin)
    P2 = spring_relax_weighted(P1, fixed_mask=fixed, k_node=k_node, iters=iters, omega=omega)


    return {
        "P0": P0,
        "P": P2,
        "inside": inside,
        "interface": interface,
        "fixed": fixed,
        "dist": dist,
        "k_node": k_node,
        "params": {
            "N": N,
            "L": L,
            "center": center,
            "R": R,
            "iters": iters,
            "omega": omega,
            "b": b,
            "k0": k0,
            "kmin": kmin,
        },
    }



if __name__ == "__main__":
    coor, x, y = make_grid_nodes(N=4, L=1.0)
    print(coor[:, 0, 0])
    print(coor[:, 1, 0])


    nx = 8
    ny = 8
    g = 1
    print("before communicator")
    comm = muGrid.Communicator()
    print("after communicator")
    print("before decomp")


    decomp = muGrid.CartesianDecomposition(
        communicator=comm,
        nb_domain_grid_pts=(nx, ny),
        nb_subdivisions=(1, 1),
        nb_ghosts_left=(g, g),
        nb_ghosts_right=(g, g),
    )
    print("after decomp")


    field = decomp.real_field("displacement", components=(3,))
    print(comm)
    print(field)
    print(type(field))
    print([name for name in dir(field) if not name.startswith("_")])


'''