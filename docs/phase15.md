# Phase 15 – Sphere drag resolved, least-squares gradients, GPU face assembly

## Sphere drag (Re = 100)
- `forces.csv` now also has the pressure (Cpx, Cpy, Cpz) and friction (Cfx, Cfy, Cfz) parts
- Diagnosis on the converged 37k-cell solution (Green-Gauss): Cp = 0.572, Cf = 0.609; the
  friction part is right, the **pressure drag** is ~20 % high. The wall evaluation is not the
  cause (cell-value pressure 0.569, wall-normal velocity in the shear 0.605 vs 0.609)
- Cause: **Green-Gauss gradients with linear face interpolation are inconsistent on skewed
  tetrahedra** (they enter the pressure gradient of the momentum equation / Rhie-Chow, linear
  upwind and the non-orthogonal correction)
- Fix: weighted least-squares gradients (`solver.gradient: "least_squares"`), exact for linear
  fields on any cell shape

| sphere mesh | cells | gradients | Cd | Cp | Cf |
|---|---|---|---|---|---|
| near 0.06 | 37 181 | Gauss (limited / unlimited LUD) | 1.187 / 1.182 | 0.572 | 0.609 |
| near 0.06 | 37 181 | **least squares** (limited LUD) | **1.0864** | 0.504 | 0.582 |
| near 0.1 (example) | 22 696 | least squares | 1.0748 | 0.505 | 0.570 |
| Johnson & Patel 1999 | | | 1.087 | | |

  (Earlier results: domain size and limiter had been ruled out, Phase 14.)
- Schäfer–Turek 2D-1 cylinder: Gauss 5.5876, least squares 5.5610 (reference 5.5795), both
  within 0.35 %; Green-Gauss stays the default (validated results unchanged), least squares
  is recommended for tetrahedral 3D meshes and used in `examples/sphere_3d.json`
- The scheme also applies to the scalar equations (k, ω, energy, scalar transport)
- Tests: least-squares gradient exact for a linear field on a Gmsh tet mesh (Green-Gauss
  error > 0.1 %); SST channel identical with both schemes on an orthogonal mesh; GUI
  "Gradients" selector (round-trip test)

## Sphere at Re = 20 / 40 (37k cells, least squares, limited LUD, 450 iterations)

| Re | Cd | Cp | Cf | Schiller–Naumann | Clift–Grace–Weber |
|---|---|---|---|---|---|
| 20 | 2.590 | 0.942 | 1.647 | 2.611 | 2.735 |
| 40 | 1.742 | 0.691 | 1.051 | 1.734 | 1.788 |
| 100 | 1.086 | 0.504 | 0.582 | 1.087 | 1.087 |

- Re 40 / 100 within 3 % of both correlations; Re 20 is 5 % below Clift–Grace–Weber
  (the Schiller–Naumann scatter is about ±5 %)

## GPU face assembly
- `gpu::assemble_convection_diffusion`: implicit upwind convection + diffusion coefficients
  and flux divergence of all internal faces (CUDA kernel with atomic accumulation, OpenMP
  target offload, deterministic serial host loop)
- Used by `solve_scalar` (energy, k, ω, VOF-related scalars) and by the pressure equation of
  the flow solver (convection factor 0: Laplacian + div(phiHbyA)) when the GPU solvers are
  selected (`solver.device: "gpu"`)
- Together with the persistent device system (Phase 14): pattern, AMG coarse levels and
  vectors on the device, CSR values assembled on the device
- Still on the CPU: the explicit parts (linear-upwind / non-orthogonal corrections, Rhie-Chow
  interpolation, momentum matrix with limiter), gradients
- Tests: kernel = CPU assembly; scalar transport with GPU settings = CPU; lid-driven cavity
  with GPU pressure assembly + AMG-PCG = CPU (1e-5); CUDA / offload builds in CI

## Definition of Done
- [x] sphere drag discrepancy found and fixed (least-squares gradients)
- [x] pressure / friction force split
- [x] GPU face assembly (scalar transport, pressure equation) + tests
- [ ] GPU run on real hardware (manual, `docs/windows_test.md` §5)
