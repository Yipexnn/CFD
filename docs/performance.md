# Performance phase – AMG pressure solver

## Changes
- **AMG-preconditioned CG** (`src/fv/amg.*`)
  - smoothed-aggregation AMG: Vanek greedy aggregation, Jacobi-smoothed prolongation, symmetric Gauss-Seidel V-cycle, LDLT on the coarsest level
  - default pressure solver (`LinearSolverType::AmgCg`)
  - the hierarchy is reused across pressure solves and rebuilt when stale (age ≥ 20 or CG iterations double)
- **CSR pattern cache** in `FvGeometry`: LDU → CSR conversion without triplet sorting
- **Limited linear upwind** (`limited_linear_upwind`): Barth–Jespersen limited reconstruction, bounded and robust at high cell Péclet numbers
- **Per-face boundary velocities** (`FlowSolver::set_boundary_velocity_profile`), e.g. parabolic inlets

## Results (4 threads)
| Case | Before | After |
|---|---|---|
| Cavity 128² (16k cells), ms / SIMPLE iteration | 79 | 15.7 |
| Cavity 256² (65k cells), ms / SIMPLE iteration | – | 72.5 (linear scaling) |
| Pressure CG iterations per solve | ~68 (IC, grows with mesh) | 9–10 (independent of mesh size) |
| Full unit-test suite | > 10 s | ~3 s |

## Validation: laminar backward-facing step (Gartling 1990, Re = 800, ER = 2)
Program: `validation/bfs_gartling.cpp` (`bfs_gartling [nx] [ny] [max_iterations]`), limited linear upwind, α_U = 0.5, α_p = 0.2.

| Mesh | Lower reattachment | Upper separation | Upper reattachment | Time |
|---|---|---|---|---|
| 300 × 40 (12k cells) | 5.88 (−3.6 %) | 4.64 (−4.3 %) | 10.41 (−0.7 %) | 78 s |
| 600 × 80 (48k cells) | 6.03 (−1.1 %) | 4.79 (−1.3 %) | 10.47 (−0.1 %) | 586 s |
| Gartling (1990) | 6.10 | 4.85 | 10.48 | |

The results converge towards the reference with grid refinement.

## Notes
- Unlimited linear upwind did not converge for this case (wiggles at cell Péclet ≈ 120). Use `limited_linear_upwind` and lower relaxation for high-Re cases on coarse meshes.
- The turbulent backward-facing step (Driver & Seegmiller) is still open: it needs a turbulent inflow boundary layer profile.
