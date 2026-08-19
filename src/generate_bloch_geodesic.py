"""Solve and plot a boundary-value geodesic for the optical Bloch manifold."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

from optical_bloch_geometry import evaluate, stationary_state


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAMMA = 1.0
# Internal coordinate order is (g/Gamma, omega/Gamma).
START = np.array([0.0, 0.0])
END = np.array([1.0, 0.4])

mpl.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "mathtext.fontset": "stix",
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    }
)


def connection(point: np.ndarray) -> np.ndarray:
    """Return Gamma^alpha_mu_nu at one control point."""

    return evaluate(point, gamma=GAMMA).christoffel


def geodesic_rhs(_s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """First-order form of ddot(lambda)^a + Gamma^a_mn v^m v^n=0."""

    rhs = np.empty_like(y)
    rhs[:2] = y[2:]
    for index in range(y.shape[1]):
        gamma = connection(y[:2, index])
        velocity = y[2:, index]
        rhs[2:, index] = -np.einsum("amn,m,n->a", gamma, velocity, velocity)
    return rhs


def boundary_conditions(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
    return np.concatenate((ya[:2] - START, yb[:2] - END))


def solve_geodesic(n_mesh: int = 80):
    """Solve the two-point geodesic boundary-value problem."""

    mesh = np.linspace(0.0, 1.0, n_mesh)
    guess = np.empty((4, n_mesh))
    guess[:2] = START[:, None] + (END - START)[:, None] * mesh
    guess[2:] = (END - START)[:, None]
    solution = solve_bvp(
        geodesic_rhs,
        boundary_conditions,
        mesh,
        guess,
        tol=1.0e-8,
        max_nodes=2000,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError(f"Geodesic solve failed: {solution.message}")
    return solution


def path_length(points: np.ndarray, velocities: np.ndarray, parameter: np.ndarray) -> float:
    speed = np.empty(parameter.size)
    for index in range(parameter.size):
        metric = evaluate(points[:, index], gamma=GAMMA).metric
        speed[index] = np.sqrt(velocities[:, index] @ metric @ velocities[:, index])
    return float(np.trapezoid(speed, parameter))


def main() -> None:
    solution = solve_geodesic()
    parameter = np.linspace(0.0, 1.0, 401)
    trajectory = solution.sol(parameter)
    geodesic = trajectory[:2]
    velocity = trajectory[2:]

    straight = START[:, None] + (END - START)[:, None] * parameter
    straight_velocity = np.repeat((END - START)[:, None], parameter.size, axis=1)
    geodesic_length = path_length(geodesic, velocity, parameter)
    straight_length = path_length(straight, straight_velocity, parameter)
    residual = float(np.max(solution.rms_residuals))

    geodesic_states = np.column_stack(
        [stationary_state(geodesic[:, i], GAMMA) for i in range(parameter.size)]
    )
    straight_states = np.column_stack(
        [stationary_state(straight[:, i], GAMMA) for i in range(parameter.size)]
    )

    fig = plt.figure(figsize=(8.2, 3.65), constrained_layout=True)
    ax_control = fig.add_subplot(1, 2, 1)
    ax_state = fig.add_subplot(1, 2, 2, projection="3d")

    # Display the control plane in the order requested in the manuscript: (omega,g).
    ax_control.plot(
        straight[1], straight[0], "--", color="0.55", linewidth=1.4,
        label="linear interpolation",
    )
    ax_control.plot(
        geodesic[1], geodesic[0], color="#B33A3A", linewidth=2.2,
        label="Levi--Civita geodesic",
    )
    ax_control.scatter([START[1], END[1]], [START[0], END[0]], c=["black", "#B33A3A"], s=28, zorder=4)
    ax_control.annotate(r"$p$", (START[1], START[0]), xytext=(5, 5), textcoords="offset points")
    ax_control.annotate(r"$q$", (END[1], END[0]), xytext=(5, 4), textcoords="offset points")
    ax_control.set(
        xlabel=r"$\omega/\Gamma$",
        ylabel=r"$g/\Gamma$",
        title="Control-manifold trajectory",
    )
    ax_control.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.04, 0.85),
        borderaxespad=0.0,
    )
    ax_control.text(0.02, 0.98, "(a)", transform=ax_control.transAxes, va="top", fontweight="bold")

    ax_state.plot(*straight_states, "--", color="0.55", linewidth=1.3)
    ax_state.plot(*geodesic_states, color="#2457A7", linewidth=2.2)
    ax_state.scatter(*geodesic_states[:, [0, -1]], c=["black", "#B33A3A"], s=26)
    ax_state.set(
        xlabel=r"$r_x$",
        ylabel=r"$r_y$",
        zlabel=r"$r_z$",
        title="Induced stationary-state path",
        xlim=(-0.50, 0.02),
    )
    # Rotate the view counterclockwise so the vertical axis is anchored on the
    # r_x=-0.5 side, and pull its label inward to avoid export-time clipping.
    ax_state.view_init(elev=23, azim=35)
    ax_state.zaxis.labelpad = -2
    ax_state.text2D(0.02, 0.98, "(b)", transform=ax_state.transAxes, va="top", fontweight="bold")

    stem = OUTPUT_DIR / "optical_bloch_geodesic"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    print(f"start (omega,g) = ({START[1]:.6f}, {START[0]:.6f})")
    print(f"end   (omega,g) = ({END[1]:.6f}, {END[0]:.6f})")
    print(f"geodesic length = {geodesic_length:.10f}")
    print(f"linear-path length = {straight_length:.10f}")
    print(f"length reduction = {straight_length - geodesic_length:.10e}")
    print(f"maximum BVP rms residual = {residual:.3e}")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
