"""Reference geometry for the amplitude-damped optical Bloch example.

Adapted from the validated implementation developed in the Duhamel_Transport
project on branch ``codex/bloch-geometry-autodiff``. It constructs the paper's
fixed-Gamma, two-control geometry symbolically and provides an optional JAX
route that differentiates through the constrained stationary-state solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class GeometryResult:
    """Geometry evaluated at one control point (g, omega)."""

    controls: np.ndarray
    state: np.ndarray
    generator: np.ndarray
    metric: np.ndarray
    two_form: np.ndarray
    christoffel: np.ndarray


def stationary_state(controls, gamma: float = 1.0) -> np.ndarray:
    """Return the exact stationary Bloch vector."""

    coupling, detuning = np.asarray(controls, dtype=float)
    denominator = gamma**2 + 2.0 * coupling**2 + 4.0 * detuning**2
    return np.array(
        [
            -4.0 * coupling * detuning / denominator,
            2.0 * gamma * coupling / denominator,
            -(gamma**2 + 4.0 * detuning**2) / denominator,
        ]
    )


def generator_coordinates(controls) -> np.ndarray:
    """Return h=(g,0,omega), the Hamiltonian generator coordinates."""

    coupling, detuning = np.asarray(controls, dtype=float)
    return np.array([coupling, 0.0, detuning])


@cache
def symbolic_geometry():
    """Return exact symbolic state, generator, G, F, and Christoffel symbols."""

    coupling, detuning, gamma = sp.symbols(
        "g omega Gamma", positive=True, real=True
    )
    coordinates = (coupling, detuning)
    denominator = gamma**2 + 2 * coupling**2 + 4 * detuning**2
    state = sp.Matrix(
        [
            -4 * coupling * detuning / denominator,
            2 * gamma * coupling / denominator,
            -(gamma**2 + 4 * detuning**2) / denominator,
        ]
    )
    generator = sp.Matrix([coupling, 0, detuning])
    dstate = state.jacobian(coordinates)
    dgenerator = generator.jacobian(coordinates)
    metric = sp.simplify(dstate.T * dstate + dgenerator.T * dgenerator)
    two_form = sp.simplify(dstate.T * dgenerator - dgenerator.T * dstate)
    inverse_metric = sp.simplify(metric.inv())

    christoffel = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    for alpha in range(2):
        for mu in range(2):
            for nu in range(2):
                value = 0
                for beta in range(2):
                    value += inverse_metric[alpha, beta] * (
                        sp.diff(metric[beta, nu], coordinates[mu])
                        + sp.diff(metric[beta, mu], coordinates[nu])
                        - sp.diff(metric[mu, nu], coordinates[beta])
                    )
                christoffel[alpha, mu, nu] = sp.factor(value / 2)

    return {
        "symbols": (coupling, detuning, gamma),
        "state": state,
        "generator": generator,
        "metric": metric,
        "two_form": two_form,
        "christoffel": christoffel,
        "kappa": sp.factor(two_form[0, 1] ** 2 / metric.det()),
    }


@cache
def _numeric_functions():
    geometry = symbolic_geometry()
    symbols = geometry["symbols"]
    return {
        name: sp.lambdify(symbols, geometry[name].tolist(), "numpy")
        for name in ("metric", "two_form", "christoffel")
    }


def evaluate(controls, gamma: float = 1.0) -> GeometryResult:
    """Evaluate G, F, and Gamma at (g, omega) for a fixed damping rate."""

    coupling, detuning = np.asarray(controls, dtype=float)
    functions = _numeric_functions()
    args = (coupling, detuning, gamma)
    return GeometryResult(
        controls=np.array([coupling, detuning]),
        state=stationary_state((coupling, detuning), gamma),
        generator=generator_coordinates((coupling, detuning)),
        metric=np.asarray(functions["metric"](*args), dtype=float),
        two_form=np.asarray(functions["two_form"](*args), dtype=float),
        christoffel=np.asarray(functions["christoffel"](*args), dtype=float),
    )


def finite_difference_geometry(
    controls, gamma: float = 1.0, step: float = 1.0e-4
) -> GeometryResult:
    """Independent central-difference check of the two-control geometry."""

    controls = np.asarray(controls, dtype=float).reshape((2,))

    def jacobian(function, point):
        columns = []
        for mu in range(2):
            displacement = np.zeros(2)
            displacement[mu] = step * max(1.0, abs(point[mu]))
            columns.append(
                (function(point + displacement) - function(point - displacement))
                / (2.0 * displacement[mu])
            )
        return np.column_stack(columns)

    def metric_map(point):
        dstate = jacobian(lambda x: stationary_state(x, gamma), point)
        dgenerator = jacobian(generator_coordinates, point)
        return dstate.T @ dstate + dgenerator.T @ dgenerator

    dstate = jacobian(lambda x: stationary_state(x, gamma), controls)
    dgenerator = jacobian(generator_coordinates, controls)
    metric = dstate.T @ dstate + dgenerator.T @ dgenerator
    two_form = dstate.T @ dgenerator - dgenerator.T @ dstate
    dmetric = np.empty((2, 2, 2))
    for eta in range(2):
        displacement = np.zeros(2)
        displacement[eta] = step * max(1.0, abs(controls[eta]))
        dmetric[eta] = (
            metric_map(controls + displacement) - metric_map(controls - displacement)
        ) / (2.0 * displacement[eta])

    inverse_metric = np.linalg.inv(metric)
    christoffel = np.empty((2, 2, 2))
    for alpha in range(2):
        for mu in range(2):
            for nu in range(2):
                christoffel[alpha, mu, nu] = 0.5 * sum(
                    inverse_metric[alpha, beta]
                    * (
                        dmetric[mu, beta, nu]
                        + dmetric[nu, beta, mu]
                        - dmetric[beta, mu, nu]
                    )
                    for beta in range(2)
                )

    return GeometryResult(
        controls=controls,
        state=stationary_state(controls, gamma),
        generator=generator_coordinates(controls),
        metric=metric,
        two_form=two_form,
        christoffel=christoffel,
    )


def validation_summary(points=None, gamma: float = 1.0) -> list[dict[str, float]]:
    """Return reproducible symbolic/FD/AD errors for the manuscript points."""

    if points is None:
        points = ((0.35, 0.12), (1.0 / np.sqrt(2.0), 0.0), (1.20, -0.40))
    rows = []
    for point in points:
        symbolic = evaluate(point, gamma)
        finite = finite_difference_geometry(point, gamma)
        autodiff = autodiff_liouvillian_geometry(point, gamma)
        rows.append(
            {
                "g": float(point[0]),
                "omega": float(point[1]),
                "ad_state": float(np.max(np.abs(autodiff.state - symbolic.state))),
                "ad_metric": float(np.max(np.abs(autodiff.metric - symbolic.metric))),
                "ad_two_form": float(
                    np.max(np.abs(autodiff.two_form - symbolic.two_form))
                ),
                "ad_christoffel": float(
                    np.max(np.abs(autodiff.christoffel - symbolic.christoffel))
                ),
                "fd_christoffel": float(
                    np.max(np.abs(finite.christoffel - symbolic.christoffel))
                ),
                "min_metric_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(autodiff.metric))
                ),
                "connection_torsion": float(
                    np.max(
                        np.abs(
                            autodiff.christoffel
                            - np.swapaxes(autodiff.christoffel, 1, 2)
                        )
                    )
                ),
            }
        )
    return rows


def autodiff_liouvillian_geometry(controls, gamma: float = 1.0) -> GeometryResult:
    """Compute r, G, F, and Gamma by differentiating the Liouvillian solve.

    JAX is optional; the symbolic route above remains the reference backend.
    The damping rate is held fixed so the differentiated controls are
    (g, omega), matching the two-dimensional manifold in the paper.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError("JAX is required for the autodiff backend") from exc

    jax.config.update("jax_enable_x64", True)
    controls_jax = jnp.asarray(controls, dtype=jnp.float64).reshape((2,))
    gamma_jax = jnp.asarray(gamma, dtype=jnp.float64)

    def liouvillian(x):
        coupling, detuning = x
        hamiltonian = jnp.asarray(
            [[0.0, coupling / 2.0], [coupling / 2.0, detuning]],
            dtype=jnp.complex128,
        )
        collapse = jnp.sqrt(gamma_jax) * jnp.asarray(
            [[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128
        )
        cdc = collapse.conj().T @ collapse
        basis = jnp.eye(4, dtype=jnp.complex128).reshape((4, 2, 2))

        def rhs(rho):
            coherent = -1.0j * (hamiltonian @ rho - rho @ hamiltonian)
            dissipative = collapse @ rho @ collapse.conj().T
            dissipative -= 0.5 * (cdc @ rho + rho @ cdc)
            return coherent + dissipative

        return jnp.stack([rhs(element).reshape(-1) for element in basis], axis=1)

    def density(x):
        generator = liouvillian(x)
        trace_row = jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex128)
        constrained = generator.at[-1, :].set(trace_row)
        target = jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=jnp.complex128)
        return jnp.linalg.solve(constrained, target).reshape((2, 2))

    def state_map(x):
        rho = density(x)
        return jnp.asarray(
            [
                2.0 * jnp.real(rho[0, 1]),
                2.0 * jnp.imag(rho[0, 1]),
                jnp.real(rho[1, 1] - rho[0, 0]),
            ]
        )

    def generator_map(x):
        return jnp.asarray([x[0], 0.0, x[1]])

    def metric_map(x):
        dstate = jax.jacfwd(state_map)(x)
        dgenerator = jax.jacfwd(generator_map)(x)
        return dstate.T @ dstate + dgenerator.T @ dgenerator

    def two_form_map(x):
        dstate = jax.jacfwd(state_map)(x)
        dgenerator = jax.jacfwd(generator_map)(x)
        return dstate.T @ dgenerator - dgenerator.T @ dstate

    metric = metric_map(controls_jax)
    # jacfwd places the differentiation index last: dmetric[mu,nu,eta].
    raw_dmetric = jax.jacfwd(metric_map)(controls_jax)
    dmetric = jnp.moveaxis(raw_dmetric, -1, 0)
    inverse_metric = jnp.linalg.inv(metric)
    christoffel = jnp.zeros((2, 2, 2), dtype=jnp.float64)
    for alpha in range(2):
        for mu in range(2):
            for nu in range(2):
                value = 0.0
                for beta in range(2):
                    value += inverse_metric[alpha, beta] * (
                        dmetric[mu, beta, nu]
                        + dmetric[nu, beta, mu]
                        - dmetric[beta, mu, nu]
                    )
                christoffel = christoffel.at[alpha, mu, nu].set(value / 2.0)

    return GeometryResult(
        controls=np.asarray(controls_jax),
        state=np.asarray(state_map(controls_jax)),
        generator=np.asarray(generator_map(controls_jax)),
        metric=np.asarray(metric),
        two_form=np.asarray(two_form_map(controls_jax)),
        christoffel=np.asarray(christoffel),
    )


if __name__ == "__main__":
    point = (1.0 / np.sqrt(2.0), 0.0)
    result = evaluate(point, gamma=1.0)
    print("controls (g, omega):", result.controls)
    print("G =\n", result.metric)
    print("F =\n", result.two_form)
    print("Gamma^alpha_mu_nu =\n", result.christoffel)
    try:
        autodiff = autodiff_liouvillian_geometry(point, gamma=1.0)
    except ImportError:
        print("JAX not installed; skipped autodiff comparison")
    else:
        print(
            "max |Gamma_symbolic-Gamma_autodiff| =",
            np.max(np.abs(result.christoffel - autodiff.christoffel)),
        )
        print("validation summary:")
        for row in validation_summary():
            print(row)
