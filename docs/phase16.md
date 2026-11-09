# Phase 16 – Automatic CPU/GPU selection, FV kernels on the GPU, sphere validation

## CPU or GPU, chosen by the system
- `solver.device`: `"auto"` (new default) uses the GPU solvers if the build finds an
  accelerator (CUDA device, OpenMP offload device), else the CPU solvers; `"cpu"` / `"gpu"`
  force one. `cfd run` prints the choice, e.g. `Linear solvers: CPU  (solver.device "auto")`
- GUI "Linear solvers": Auto (shows "GPU found" / "no GPU: CPU"), CPU, GPU
- MPI runs always use the CPU solvers
- Tests: default is auto, auto → GPU only with a device, round trip keeps auto, GUI round trip

## Finite-volume kernels on the GPU
- `gpu::DeviceFv`: mesh (faces, cell-face lists, geometry, least-squares matrices) copied to the
  device once; each call moves only the fields
- On the device: Green-Gauss and least-squares gradients (scalar, velocity, pressure with the
  tangential boundary extrapolation), Barth-Jespersen limiter, explicit face fluxes of the
  scalar equations (linear upwind, non-orthogonal correction) and of the momentum equation
  (linear upwind with limiter, non-orthogonal correction, transpose term), plus the matrix
  assembly (Phase 15) and the linear solvers (Phase 14)
- Each cell gathers over its faces in face order: no atomics, deterministic, same result as
  the CPU face loops
- One source for the per-cell math (`fv_cell.inc`) used by the CUDA kernels, OpenMP target
  offload and host threads (forced `"gpu"` without a device)
- Used with the GPU solvers on serial meshes; the CPU path is unchanged
- Still on the CPU: Rhie-Chow face fluxes, boundary conditions, turbulence source terms
- Tests: device gradients (Gauss, least squares) and scalar corrections = CPU (1e-12) on a
  skewed mesh; skewed lid-driven cavity with limited linear upwind, least squares and variable
  viscosity: GPU path = CPU (1e-5); CUDA and OpenMP-offload builds in CI

## Mesh refinement boxes
- `mesh.refine_boxes`: `[{"min": [...], "max": [...], "size": h}]` – Gmsh box fields combined
  with the body refinement (e.g. wakes); test: smaller cells in the box, case round trip

## Sphere at Re = 20
| run (300–450 iterations, converged; least squares, limited LUD) | cells | Cd | Cp | Cf |
|---|---|---|---|---|
| base (domain −5…15 × ±5 D, far size 1.0) | 37 181 | 2.590 | 0.942 | 1.647 |
| finer wall (9 layers, first 0.004) | 42 879 | 2.543 | 0.927 | 1.616 |
| inlet 10 D upstream | 41 235 | 2.572 | 0.929 | 1.643 |
| lateral ±10 D | 81 661 | 2.558 | 0.923 | 1.635 |
| finer far field (size 0.6, refine distance 4) | 90 569 | 2.613 | 0.965 | 1.648 |
| Schiller–Naumann / Clift–Grace–Weber | | 2.611 / 2.735 | | |
| numerical references (Magnaudet et al. / Tabata & Itakura) | | 2.707 / 2.724 | | |

- Larger domains lower Cd slightly, a finer far field raises it; the result stays 4–6 % below
  the accurate numerical references at Re 20 (Re 40: −3 %, Re 100: within 1 %)
- Open: the cause of the low-Re deficit (both pressure and friction parts are low)

## Sphere at Re = 300 (unsteady wake)
- Pending
