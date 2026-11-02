# Phase 2 – Laminar incompressible solver

Goal: steady (SIMPLE) and transient (PISO) incompressible laminar Navier–Stokes on the Phase 1 mesh, validated against analytic and benchmark data.

## Design

New module `cfd::fv` (`src/fv`, static library `cfd_fv`, depends on `cfd::mesh`).

**Equations** (kinematic pressure p/ρ, constant ν):
- ∂U/∂t + ∇·(φU) − ∇·(ν∇U) = −∇p
- ∇·U = 0

**Fields**: `ScalarField` / `VectorField` = cell values plus one boundary value per boundary face.

**Boundary conditions** (per patch):
| Kind | U | p |
|---|---|---|
| `velocity_inlet` | fixed value | zero gradient |
| `wall` (no-slip / moving) | fixed value (wall velocity) | zero gradient |
| `pressure_outlet` | zero gradient | fixed value |
| `symmetry` | normal component zero, tangential zero gradient | zero gradient |

**Discretisation**
- Collocated cell-centred finite volume
- Face interpolation: distance-weighted linear
- Gradient: Gauss linear
- Convection: upwind, or linear-upwind via deferred correction (default)
- Diffusion: orthogonal part implicit (|Sf|/|d|·Δ), non-orthogonal correction explicit (over-relaxed)
- Time: implicit Euler

**Matrix**: LDU storage (diag, lower/upper per internal face, source). Conversion to `Eigen::SparseMatrix<RowMajor>`. Solvers: Eigen `ConjugateGradient` (pressure, symmetric) and `BiCGSTAB` (momentum), with diagonal preconditioner and configurable tolerance/max iterations. AMG comes later for performance.

**Pressure–velocity coupling** (OpenFOAM-style HbyA formulation, which includes Rhie–Chow):
1. Momentum predictor with under-relaxation α_U
2. rAU = 1/a_P, HbyA = H(U)/a_P, φ_HbyA = (HbyA)_f · Sf
3. Pressure: ∇·(rAU_f ∇p) = ∇·φ_HbyA, with non-orthogonal correctors
4. φ = φ_HbyA − pressure flux; U = HbyA − rAU ∇p; p under-relaxation α_p (SIMPLE)
- SIMPLE (steady): loop until residuals < tol
- PISO (transient): per time step, momentum predictor + n pressure correctors
- Pressure reference cell when no fixed-pressure boundary

**API**: `Case` (mesh, ν, BCs per patch name, solver settings) → `SimpleSolver` / `PisoSolver` with `iterate()` / `run()` and a residual history.

**Output**: legacy VTK writer (`write_vtk`) for quick ParaView checks (full VTU/case I/O is Phase 4).

**CLI**: `cfd run-cavity <n> <Re>` (lid-driven cavity demo, writes `cavity.vtk`).

## Validation
- **Poiseuille channel** (plates, periodic-like long channel with parabolic outlet check): max U = 1.5·U_mean, Δp = 12 ν U_mean L / h²; error < 2 %
- **Lid-driven cavity Re = 100** (n×n×1 cells, front/back symmetry): centreline u vs. Ghia et al. (1982); max deviation < 0.03 on 32×32
- **Conservation**: Σ φ over each cell ~ 0 after convergence
- **Transient**: PISO for impulsively started channel flow converges to the steady Poiseuille profile
- Cylinder flow: deferred to Phase 6 (needs auto meshing)

## Tasks
| # | Item | Who |
|---|---|---|
| 1 | `fields.hpp` (fields + BC types) | O |
| 2 | `ldu_matrix.hpp/.cpp` (assembly helpers, Eigen conversion, solve) | O |
| 3 | `operators.hpp/.cpp` (interpolation weights, Gauss gradient, convection/diffusion assembly) | O |
| 4 | `simple_solver.hpp/.cpp`, PISO | O |
| 5 | `vtk_writer.hpp/.cpp` (legacy ASCII unstructured grid, cell data) | H |
| 6 | Ghia reference data header (Re 100 centreline) | H |
| 7 | tests: LDU matrix / linear solvers | H |
| 8 | tests: gradient exactness (linear field), Laplacian | O |
| 9 | validation tests: Poiseuille, cavity, PISO | O |
| 10 | CLI `run-cavity` | H |

## Definition of Done
- [x] CI green
- [x] Pressure gradient of a linear field is exact (checked on sheared mesh)
- [x] Poiseuille and cavity validations pass within tolerance
- [x] `cfd run-cavity 32 100` converges and writes VTK
