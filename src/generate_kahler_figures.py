"""Generate individual state-generator Kähler compatibility figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "mathtext.fontset": "stix",
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    plt.close(fig)


# Symbolic state-generator geometry for an amplitude-damped two-level system.
g, omega, gamma = sp.symbols("g omega Gamma", real=True, positive=True)
denominator = gamma**2 + 2 * g**2 + 4 * omega**2

state = sp.Matrix(
    [
        -4 * g * omega / denominator,
        2 * gamma * g / denominator,
        -(gamma**2 + 4 * omega**2) / denominator,
    ]
)
generator = sp.Matrix([g, 0, omega])
parameters = [g, omega]

dstate = [sp.simplify(state.diff(parameter)) for parameter in parameters]
dgenerator = [sp.simplify(generator.diff(parameter)) for parameter in parameters]

metric = sp.Matrix(
    2,
    2,
    lambda i, j: sp.simplify(
        dstate[i].dot(dstate[j]) + dgenerator[i].dot(dgenerator[j])
    ),
)
two_form = sp.Matrix(
    2,
    2,
    lambda i, j: sp.simplify(
        dstate[i].dot(dgenerator[j]) - dstate[j].dot(dgenerator[i])
    ),
)

f_gomega = sp.factor(two_form[0, 1])
det_metric = sp.factor(metric.det())
kappa = sp.factor(f_gomega**2 / det_metric)
compatibility_defect = sp.factor(det_metric - f_gomega**2)

kappa_function = sp.lambdify((g, omega, gamma), kappa, "numpy")
defect_function = sp.lambdify((g, omega, gamma), compatibility_defect, "numpy")

# Dimensionless scan with Gamma = 1.
gamma_value = 1.0
g_values = np.linspace(0.0, 2.0, 500)
omega_values = np.linspace(-1.5, 1.5, 500)
g_grid, omega_grid = np.meshgrid(g_values, omega_values)

kappa_grid = np.asarray(
    np.real_if_close(kappa_function(g_grid, omega_grid, gamma_value)), dtype=float
)
defect_grid = np.asarray(
    np.real_if_close(defect_function(g_grid, omega_grid, gamma_value)), dtype=float
)

compatible_g = 1.0 / np.sqrt(2.0)

# Figure 1: Kähler compatibility ratio.
fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
image = ax.pcolormesh(
    g_grid,
    omega_grid,
    kappa_grid,
    shading="auto",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    rasterized=True,
)
colorbar = fig.colorbar(image, ax=ax, pad=0.025)
colorbar.set_label(r"$\kappa=F_{g\omega}^{,2}/\det G$")
ax.plot(
    compatible_g,
    0.0,
    marker="o",
    markersize=4,
    markerfacecolor="black",
    markeredgecolor="black",
    markeredgewidth=0.6,
)
ax.set(xlabel=r"$g/\Gamma$", ylabel=r"$\omega/\Gamma$")
ax.set_title(r"Kähler compatibility ratio")
ax.text(
    0.02,
    0.96,
    "(a)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    color="white",
    fontweight="bold",
    fontsize=13,
)
save_figure(fig, "kahler_compatibility_ratio")

# Figure 2: logarithmic compatibility defect.
fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
log_defect = np.log10(np.abs(defect_grid) + 1.0e-12)
image = ax.pcolormesh(
    g_grid,
    omega_grid,
    log_defect,
    shading="auto",
    cmap="magma",
    rasterized=True,
)
colorbar = fig.colorbar(image, ax=ax, pad=0.025)
colorbar.set_label(r"$\log_{10}(\det G-F_{g\omega}^{,2})$")
ax.plot(
    compatible_g,
    0.0,
    marker="o",
    markersize=7,
    markerfacecolor="none",
    markeredgecolor="white",
    markeredgewidth=1.6,
)
ax.set(xlabel=r"$g/\Gamma$", ylabel=r"$\omega/\Gamma$")
ax.set_title(r"Kähler compatibility defect")
save_figure(fig, "kahler_compatibility_defect")

# Figure 3: resonant cut.
g_line = np.linspace(0.001, 2.0, 1000)
kappa_line = np.asarray(
    np.real_if_close(kappa_function(g_line, 0.0, gamma_value)), dtype=float
)

# Match the near-square canvas used by panel (a) so both panels have the same
# displayed height when included at equal width in the manuscript.
fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
ax.plot(g_line, kappa_line, color="#2457A7", linewidth=2.0)
ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
ax.axvline(compatible_g, color="#B33A3A", linestyle="--", linewidth=1.2)
ax.plot(
    compatible_g,
    1.0,
    marker="o",
    markersize=6,
    markerfacecolor="white",
    markeredgecolor="#B33A3A",
    markeredgewidth=1.4,
)
ax.annotate(
    r"$g/\Gamma=1/\sqrt{2}$",
    xy=(compatible_g, 1.0),
    xytext=(compatible_g + 0.07, 0.60),
    textcoords="data",
    color="#8E2C2C",
)
ax.set(xlabel=r"$g/\Gamma$", ylabel=r"$\kappa$", ylim=(0.0, 1.05))
ax.set_title(r"Resonant cut: $\omega=0$")
ax.text(
    0.02,
    0.96,
    "(b)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    color="black",
    fontweight="bold",
    fontsize=13,
)
save_figure(fig, "kahler_resonant_cut")

print("Generated:")
for filename in sorted(OUTPUT_DIR.glob("kahler_*")):
    print(filename)
