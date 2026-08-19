# State--Generator Geometry of Open Quantum Systems: Compatibility and Covariant Transport

This directory is the computational source archive for the figures and
numerical checks accompanying the paper named above. The scripts in `src/`
regenerate publication-ready PNG files for inclusion in
the LaTeX manuscript and matching PDF files for vector archival and later
editing.  Generated artwork is written to the repository-level `figures/`
directory; the scripts do not write into `src/`.

## Environment

The calculations were developed with Python 3 and require:

- NumPy
- SciPy
- SymPy
- Matplotlib
- JAX with 64-bit support for the optional automatic-differentiation checks
- A working LaTeX installation for Matplotlib's LaTeX-rendered labels in the
  geodesic figure

A suitable environment can be created with either `pip` or `conda`.  For
example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The archived validation reported in the manuscript was run with JAX 0.11.0,
jaxlib 0.11.0, NumPy 2.1.3, SciPy 1.15.3, SymPy 1.13.3, and Matplotlib 3.10.0.

Run every command below from the repository root.  For a headless system, set
Matplotlib's backend and cache directory explicitly:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mpl-jcp2b
```

The pinned `requirements.txt` records the validated archive environment.
Newer dependency releases may work, but should be treated as a new
computational environment and revalidated before publication.

## Source files

### `src/optical_bloch_geometry.py`

This is the shared geometry backend for the amplitude-damped optical Bloch
model.  At fixed damping rate `Gamma`, its internal control-coordinate order is

```text
(g/Gamma, omega/Gamma)
```

The module provides:

- the exact stationary Bloch vector `stationary_state`;
- Hamiltonian generator coordinates `generator_coordinates`;
- exact symbolic construction of the induced metric `G`, two-form `F`,
  compatibility ratio `kappa`, and Christoffel symbols;
- numerical evaluation of the symbolic expressions;
- an independent central-finite-difference geometry calculation; and
- a JAX route that differentiates through the trace-constrained Liouvillian
  stationary-state solve.

Run the built-in validation summary with:

```bash
python src/optical_bloch_geometry.py
```

The script evaluates the compatible resonant point and compares the symbolic,
finite-difference, and automatic-differentiation connections at the three
control points reported in the manuscript.  If JAX is unavailable, the
symbolic geometry remains usable but the automatic-differentiation comparison
is skipped.

### `src/generate_bloch_geodesic.py`

This script generates Fig. 2 of the manuscript.  It imports the optical Bloch
geometry backend above and solves the two-point geodesic boundary-value
problem

```text
d2 lambda^a/ds2 + Gamma^a_mn(lambda) dlambda^m/ds dlambda^n/ds = 0
```

from

```text
p = (omega/Gamma, g/Gamma) = (0.00, 0.00)
q = (omega/Gamma, g/Gamma) = (0.40, 1.00).
```

The internal arrays retain `(g, omega)` ordering; the plotted control axes and
reported endpoints use `(omega, g)` ordering.  `scipy.integrate.solve_bvp`
starts from a straight-line guess and evaluates the symbolic Levi--Civita
connection at every collocation point.  The script then computes the physical
lengths of both the geodesic and straight interpolation using the induced
metric.

Run:

```bash
python src/generate_bloch_geodesic.py
```

Outputs:

```text
figures/optical_bloch_geodesic.png
figures/optical_bloch_geodesic.pdf
```

The terminal output records the endpoints, both path lengths, their
difference, and the maximum boundary-value-solver rms residual.  Edit the
`START`, `END`, or `GAMMA` constants near the top of the script to explore a
different transport problem.  The PNG is the file included by the manuscript;
the PDF is the vector archive.

### `src/generate_kahler_figures.py`

This script symbolically constructs the fixed-`Gamma` two-control optical
Bloch geometry and scans the compatibility ratio

```text
kappa = F_(g,omega)^2 / det(G).
```

It generates the two panels used in Fig. 1 as individual files, together with
an additional compatibility-defect diagnostic:

```text
figures/kahler_compatibility_ratio.{png,pdf}
figures/kahler_resonant_cut.{png,pdf}
figures/kahler_compatibility_defect.{png,pdf}
```

Run:

```bash
python src/generate_kahler_figures.py
```

The manuscript places the ratio and resonant-cut PNG files side by side.  The
panel letters are part of the generated artwork.

### `src/generate_parallel_transport_figure.py`

This script produces the conceptual curved-manifold illustration of parallel
transport.  It draws a model sphere, a path from `p` to `q`, transported
tangent vectors, and the defining equations for metric-compatible parallel
transport.

Run:

```bash
python src/generate_parallel_transport_figure.py
```

Outputs:

```text
figures/parallel_transport_manifold.png
figures/parallel_transport_manifold.pdf
```

The path and view orientation are controlled directly by constants and arrays
near the beginning of the script.

## Rebuilding the manuscript

After regenerating figures, compile `main.tex`.  A typical local command is:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The manuscript deliberately includes PNG artwork while the corresponding PDF
files remain in `figures/` for archival and publication use.  Verify both the
standalone image and the compiled manuscript after changing view angles, font
sizes, annotations, or legend placement because tight bounding boxes and 3D
axis labels can change during export.

## Reproducibility notes

- Dimensionless plots set `Gamma = 1`.
- The optical Bloch geodesic uses a solver tolerance of `1e-8` and samples the
  converged path at 401 points for plotting and length integration.
- JAX is explicitly configured for 64-bit arithmetic before differentiating
  the Liouvillian solve.
- PNG files are exported at 400 dpi; PDF files retain vector text and line
  art where supported by Matplotlib.
- Generated results should be committed together with the exact source change
  that produced them.

## Archive validation checklist

Before making a tagged release or a Zenodo deposit, run:

```bash
python src/optical_bloch_geometry.py
python src/generate_bloch_geodesic.py
git diff --exit-code -- figures/optical_bloch_geodesic.png \
  figures/optical_bloch_geodesic.pdf
```

The first command compares the symbolic connection with independent
finite-difference and JAX automatic-differentiation calculations. The second
recomputes the boundary-value geodesic and its path length. The final command
checks that the regenerated publication artifacts agree with the archived
files. PDF metadata can make a byte-for-byte comparison platform-dependent;
if only the PDF differs, inspect the rendered pages and numerical terminal
output before accepting the new artifact.

## License

The code, figures, and supporting archive are licensed under the
[Creative Commons Attribution 4.0 International License](LICENSE.md)
(CC BY 4.0). Reuse and adaptation are permitted with appropriate attribution.
