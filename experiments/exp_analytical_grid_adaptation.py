import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from muFFTTO.analytical_grid_adaptation import adapt_grid_to_circle

# Fixed parameters
DOMAIN_HALF_SIZE = 40.0
CIRCLE_CENTER = (0.0, 0.0)
CIRCLE_RADIUS = 20.0
NB_ITERS = 400
OMEGA = 0.8
K0 = 0.25
KMIN = 0.005

#coor = make_grid_nodes(N=8, L=2.0)
#print(coor)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the grid-adaptation experiment.

    The function reads optional command-line arguments specifying the mesh
    size and the stiffness-decay exponent. If no values are provided in the
    terminal, default values are used.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Namespace containing the parsed command-line arguments.
        - args.N : int
            Mesh size used for the grid adaptation.
        - args.b : float
            Stiffness-decay exponent used for the grid adaptation.
    """
    parser = argparse.ArgumentParser(
        description="Run analytical grid adaptation for one specified N and b."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=16,
        help="Grid size N (default: 16)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.5,
        help="Stiffness exponent b (default: 0.5)",
    )
    return parser.parse_args()

def plot_initial_and_adapted_mesh(
    P0: np.ndarray,
    P: np.ndarray,
    N: int,
    b: float,
    elapsed_time: float,
) -> None:
    """
    Plot the initial and adapted meshes side by side.

    The function visualizes the grid lines of the initial regular mesh and the
    adapted mesh obtained after grid deformation toward a circular geometry.
    The selected mesh size, stiffness exponent, and total runtime are shown in
    the figure title.

    Parameters
    ----------
    P0 : numpy.ndarray
        Initial nodal coordinates with shape [nx, ny, xy].
    P : numpy.ndarray
        Adapted nodal coordinates with shape [nx, ny, xy].
    N : int
        Mesh size used for the grid adaptation.
    b : float
        Stiffness-decay exponent used for the grid adaptation.
    elapsed_time : float
        Total runtime of the grid-adaptation procedure in seconds.

    Returns
    -------
    None
    """
    N_plot = P0.shape[0] - 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_init, ax_def = axes

    # Initial grid
    for i in range(N_plot + 1):
        ax_init.plot(P0[i, :, 0], P0[i, :, 1], color="0.2", linewidth=0.7)
    for j in range(N_plot + 1):
        ax_init.plot(P0[:, j, 0], P0[:, j, 1], color="0.2", linewidth=0.7)

    ax_init.set_aspect("equal", adjustable="box")
    ax_init.set_title("Initial grid")
    ax_init.set_xlabel("x")
    ax_init.set_ylabel("y")

    # Adapted grid
    for i in range(N_plot + 1):
        ax_def.plot(P[i, :, 0], P[i, :, 1], color="0.2", linewidth=0.7)
    for j in range(N_plot + 1):
        ax_def.plot(P[:, j, 0], P[:, j, 1], color="0.2", linewidth=0.7)

    ax_def.set_aspect("equal", adjustable="box")
    ax_def.set_title("Adapted grid")
    ax_def.set_xlabel("x")
    ax_def.set_ylabel("y")

    fig.suptitle(
        f"Grid adaptation to circle | N = {N}, b = {b:.2f}, time = {elapsed_time:.4f} s"
    )
    fig.tight_layout()
    plt.show()


def main() -> None:
    """
    Run one grid-adaptation experiment and display the resulting meshes.

    The function parses the command-line arguments, executes the analytical
    grid-adaptation procedure for one pair of parameters ``N`` and ``b``,
    measures the runtime, and visualizes the initial and adapted meshes.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = parse_args()

    start_time = time.perf_counter()
    result = adapt_grid_to_circle(
        N=args.N,
        L=DOMAIN_HALF_SIZE,
        center=CIRCLE_CENTER,
        R=CIRCLE_RADIUS,
        iters=NB_ITERS,
        omega=OMEGA,
        b=args.b,
        k0=K0,
        kmin=KMIN,
    )
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    plot_initial_and_adapted_mesh(
        result["P0"],
        result["P"],
        N=args.N,
        b=args.b,
        elapsed_time=elapsed_time,
    )

if __name__ == "__main__":
    main()