"""Generate a conceptual illustration of parallel transport on a manifold."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Z_ROTATION = -np.pi / 6.0  # 30 degrees clockwise as viewed from +z.

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "stix",
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    }
)


def curve(t: float) -> np.ndarray:
    """A smooth path on the unit sphere."""
    theta = 2.12 - 0.46 * t + 0.05 * np.sin(1.2 * t)
    phi = -0.78 + t + Z_ROTATION
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def curve_derivative(t: float) -> np.ndarray:
    """Analytic derivative of the spherical path."""
    theta = 2.12 - 0.46 * t + 0.05 * np.sin(1.2 * t)
    phi = -0.78 + t + Z_ROTATION
    theta_prime = -0.46 + 0.06 * np.cos(1.2 * t)
    phi_prime = 1.0
    dtheta = np.array(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ]
    )
    dphi = np.array(
        [
            -np.sin(theta) * np.sin(phi),
            np.sin(theta) * np.cos(phi),
            0.0,
        ]
    )
    return theta_prime * dtheta + phi_prime * dphi


def parallel_rhs(t: float, vector: np.ndarray) -> np.ndarray:
    """Extrinsic form of Levi-Civita parallel transport on the unit sphere."""
    position = curve(t)
    velocity = curve_derivative(t)
    return -np.dot(vector, velocity) * position


def rk4_step(t: float, vector: np.ndarray, step: float) -> np.ndarray:
    k1 = parallel_rhs(t, vector)
    k2 = parallel_rhs(t + step / 2, vector + step * k1 / 2)
    k3 = parallel_rhs(t + step / 2, vector + step * k2 / 2)
    k4 = parallel_rhs(t + step, vector + step * k3)
    return vector + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


t_values = np.linspace(0.0, 2.45, 700)
positions = np.array([curve(t) for t in t_values])

# Choose an initial tangent vector perpendicular to the path direction.
initial_position = positions[0]
initial_tangent = curve_derivative(t_values[0])
initial_tangent /= np.linalg.norm(initial_tangent)
transported = np.empty_like(positions)
transported[0] = np.cross(initial_position, initial_tangent)
transported[0] /= np.linalg.norm(transported[0])

for index in range(len(t_values) - 1):
    dt = t_values[index + 1] - t_values[index]
    transported[index + 1] = rk4_step(t_values[index], transported[index], dt)
    # Remove accumulated numerical normal component and preserve the norm.
    transported[index + 1] -= (
        np.dot(transported[index + 1], positions[index + 1]) * positions[index + 1]
    )
    transported[index + 1] /= np.linalg.norm(transported[index + 1])


fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
ax = fig.add_subplot(111, projection="3d")

# Translucent spherical model manifold.
u = np.linspace(0, 2 * np.pi, 150) + Z_ROTATION
v = np.linspace(0.18, np.pi - 0.18, 85)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(
    x,
    y,
    z,
    color="#BFD7EA",
    alpha=0.38,
    linewidth=0,
    antialiased=True,
    shade=True,
)
ax.plot_wireframe(
    x,
    y,
    z,
    rstride=18,
    cstride=14,
    color="#6E8FA8",
    linewidth=0.35,
    alpha=0.24,
)

# Transport path.
path_offset = 1.012 * positions
ax.plot(
    path_offset[:, 0],
    path_offset[:, 1],
    path_offset[:, 2],
    color="#A12D2D",
    linewidth=3.0,
    zorder=10,
)

# Parallel-transported vectors along the curve.
arrow_indices = [0, 125, 250, 390, 535, 699]
arrow_length = 0.31
for count, index in enumerate(arrow_indices):
    point = 1.025 * positions[index]
    vector = transported[index]
    color = "#163D6B" if count not in (0, len(arrow_indices) - 1) else "#092A4A"
    ax.quiver(
        point[0],
        point[1],
        point[2],
        vector[0],
        vector[1],
        vector[2],
        length=arrow_length,
        normalize=True,
        color=color,
        linewidth=2.1,
        arrow_length_ratio=0.24,
        zorder=20,
    )

# Endpoint markers and labels.
start = positions[0]
finish = positions[-1]
ax.scatter(*start, color="#111111", s=24, depthshade=False, zorder=30)
ax.scatter(*finish, color="#111111", s=24, depthshade=False, zorder=30)
ax.text(*(1.09 * start + np.array([-0.02, -0.04, 0.03])), r"$p$", fontsize=13)
ax.text(*(1.09 * finish + np.array([0.02, 0.00, 0.04])), r"$q$", fontsize=13)

middle = positions[len(positions) // 2]
ax.text(
    *(1.10 * middle + np.array([0.01, -0.02, 0.08])),
    r"$\gamma$",
    color="#8E2525",
    fontsize=15,
)

ax.text2D(
    0.05,
    0.92,
    r"$\nabla_{\dot\gamma}V=0$",
    transform=ax.transAxes,
    fontsize=16,
    color="#163D6B",
)
ax.text2D(
    0.05,
    0.84,
    r"parallel transport preserves $g(V,V)$",
    transform=ax.transAxes,
    fontsize=11,
    color="#333333",
)
ax.text2D(
    0.05,
    0.76,
    r"$ds^2=g_{\mu\nu}\,d\lambda^\mu d\lambda^\nu$",
    transform=ax.transAxes,
    fontsize=15,
    color="#333333",
)

ax.set_title("Parallel transport on a curved model manifold", pad=8)
ax.set_box_aspect((1.15, 1.05, 0.92))
ax.view_init(elev=24, azim=-55)
ax.set_axis_off()

for extension in ("pdf", "png"):
    fig.savefig(OUTPUT_DIR / f"parallel_transport_manifold.{extension}")
plt.close(fig)

print(OUTPUT_DIR / "parallel_transport_manifold.pdf")
print(OUTPUT_DIR / "parallel_transport_manifold.png")
